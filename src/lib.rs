#![feature(once_cell)]

use numpy::IntoPyArray;
use once_cell::sync::OnceCell;
use std::sync::Mutex;

use pyo3::types::PyModule;
use pyo3::{Py, PyAny, Python};

#[non_exhaustive]
pub struct Classifier {
    // predict contains model loading code so it is loaded every call.
    // Load model separately.
    predict: Mutex<Py<PyAny>>,
}

pub fn init() -> anyhow::Result<&'static Classifier> {
    static INSTANCE: OnceCell<Classifier> = OnceCell::new();
    INSTANCE.get_or_try_init(|| {
        let predict = Mutex::new(Python::with_gil(|py| {
            anyhow::Ok(
                PyModule::from_code(py, include_str!("py/predict.py"), "predict.py", "predict")?
                    .getattr("predict")?
                    .into(),
            )
        })?);
        Ok(Classifier { predict })
    })
}

// SHAPE 150x39

impl Classifier {
    pub fn train_batch(&self) -> anyhow::Result<()> {
        todo!()
    }

    pub fn predict(&self, data: ndarray::Array4<f64>) -> anyhow::Result<ndarray::Array2<f32>> {
        let predict = self.predict.lock().unwrap();

        Python::with_gil(|py| {
            let result: &numpy::PyArray2<f32> = predict
                .as_ref(py)
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
