use std::sync::Arc;

use datafusion::arrow::array::{Array, Int32Array, Int64Builder, ListArray};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{create_udf, ColumnarValue, ScalarUDF, Volatility};
use pyo3::prelude::*;

#[pyfunction]
fn list_len_py() -> usize {
    0
}

#[pymodule]
fn hailx_list(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_len_py, m)?)?;
    Ok(())
}

/// Create a DataFusion UDF that sums the elements of a list of integers, ignoring nulls.
///
/// The function accepts `List<Int32>` or `List<Nullable<Int32>>` and returns an
/// `Int64`, or `NULL` if the input list is empty or contains only nulls.
pub fn list_sum_udf() -> ScalarUDF {
    let fun = |args: &[ColumnarValue]| -> Result<ColumnarValue> {
        let array = match &args[0] {
            ColumnarValue::Array(arr) => arr.clone(),
            _ => {
                return Err(DataFusionError::Execution("expected array".into()))
            }
        };

        let list_array = array
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| DataFusionError::Execution("expected ListArray".into()))?;

        let values = list_array
            .values()
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| DataFusionError::Execution("expected Int32 values".into()))?;

        let offsets = list_array.value_offsets();
        let mut builder = Int64Builder::with_capacity(list_array.len());

        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                builder.append_null();
                continue;
            }

            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let mut sum = 0i64;
            let mut seen = false;

            for j in start..end {
                if values.is_valid(j) {
                    sum += values.value(j) as i64;
                    seen = true;
                }
            }

            if seen {
                builder.append_value(sum);
            } else {
                builder.append_null();
            }
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    };

    create_udf(
        "list_sum",
        vec![DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(fun),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Array;
    use datafusion::prelude::*;
    use vortex_array::arrays::{ListArray as VortexListArray, PrimitiveArray, StructArray};
    use vortex_array::IntoArray;
    use vortex_array::validity::Validity;
    use vortex_dtype::FieldNames;

    #[test]
    fn test_list_len_py() {
        assert_eq!(list_len_py(), 0);
    }

    #[tokio::test]
    async fn test_list_sum_udf_on_vortex_gt() {
        // Build Vortex list array representing a GT column
        let elements = PrimitiveArray::from_option_iter::<i32, _>(vec![
            Some(0),
            Some(1),
            Some(2), // row 0
            Some(2),
            Some(1), // row 1
            Some(2),
            None, // row 2
            None,
            None, // row 3
        ]);
        let offsets = PrimitiveArray::from_iter(vec![0u32, 3, 5, 7, 9]);
        let gt_list = VortexListArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();
        let names: FieldNames = vec!["gt".into()].into();
        let struct_array =
            StructArray::try_new(names, vec![gt_list], 4, Validity::NonNullable).unwrap();
        let batch = struct_array.into_record_batch().unwrap();

        // Register the batch and UDF in DataFusion
        let ctx = SessionContext::new();
        ctx.register_batch("t", batch).unwrap();
        let udf = list_sum_udf();
        ctx.register_udf(udf.clone());

        let df = ctx
            .table("t")
            .await
            .unwrap()
            .select(vec![udf.call(vec![col("gt")]).alias("s")])
            .unwrap();
        let results = df.collect().await.unwrap();

        let array = results[0]
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap();
        let actual: Vec<Option<i64>> = (0..array.len())
            .map(|i| if array.is_null(i) { None } else { Some(array.value(i)) })
            .collect();
        assert_eq!(actual, vec![Some(3), Some(3), Some(2), None]);
    }
}
