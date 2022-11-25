#[non_exhaustive]
pub struct Classifier;

static CLASSIFIER: Classifier = Classifier {};

pub fn init() -> &'static Classifier {
    &CLASSIFIER
}

// SHAPE 150x39

impl Classifier {
    pub fn train_batch(&self) -> anyhow::Result<()> {
        todo!()
    }

    pub fn predict(&self, data: ndarray::Array4<f64>) -> anyhow::Result<ndarray::Array2<f32>> {
        use numpy::IntoPyArray;
        use pyo3::prelude::*;

        Python::with_gil(|py| {
            let result: &numpy::PyArray2<f32> =
                PyModule::from_code(py, include_str!("py/predict.py"), "predict.py", "predict")?
                    .getattr("predict")?
                    .call1((data.into_pyarray(py),))?
                    .extract()?;
            Ok(result.readonly().as_array().to_owned())
        })
    }

    pub fn verify(
        &self,
        data: ndarray::Array4<f64>,
        labels: ndarray::Array1<u32>,
    ) -> anyhow::Result<f64> {
        use numpy::IntoPyArray;
        use pyo3::prelude::*;

        Python::with_gil(|py| {
            let result: f64 =
                PyModule::from_code(py, include_str!("py/verify.py"), "verify.py", "verify")?
                    .getattr("verify")?
                    .call1((data.into_pyarray(py), labels.into_pyarray(py), 3))?
                    .extract()?;
            Ok(result)
        })
    }
}
