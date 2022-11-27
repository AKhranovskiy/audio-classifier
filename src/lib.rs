#![feature(once_cell)]

use numpy::IntoPyArray;
use once_cell::sync::OnceCell;
use std::sync::Mutex;

use pyo3::types::PyModule;
use pyo3::{Py, PyAny, Python};

#[non_exhaustive]
pub struct Classifier {
    inner: Mutex<Inner>,
}

struct Inner {
    model: Py<PyAny>,
    predict: Py<PyAny>,
}

static DEF_LOAD_MODEL: &str = r"
import tensorflow as tf
def load_model(path):
    return tf.keras.models.load_model(path)
";

static DEF_PREDICT: &str = r"
def predict(model, data):
    return model(data).numpy()
";

pub fn init() -> anyhow::Result<&'static Classifier> {
    static INSTANCE: OnceCell<Classifier> = OnceCell::new();
    INSTANCE.get_or_try_init(|| {
        let model = Python::with_gil(|py| {
            let load_model =
                PyModule::from_code(py, DEF_LOAD_MODEL, "load_model.py", "load_model")?
                    .getattr("load_model")?;
            anyhow::Ok(load_model.call1(("./model",))?.into())
        })?;

        let predict = Python::with_gil(|py| {
            anyhow::Ok(
                PyModule::from_code(py, DEF_PREDICT, "predict.py", "predict")?
                    .getattr("predict")?
                    .into(),
            )
        })?;

        Ok(Classifier {
            inner: Mutex::new(Inner { model, predict }),
        })
    })
}

// SHAPE 150x39

impl Classifier {
    pub fn predict(&self, data: ndarray::Array4<f64>) -> anyhow::Result<ndarray::Array2<f32>> {
        let inner = self.inner.lock().unwrap();
        let Inner { model, predict } = &*inner;

        Python::with_gil(|py| {
            let data = data.into_pyarray(py);
            let model = model.as_ref(py);
            let predict = predict.as_ref(py);
            let pyarray: &numpy::PyArray2<f32> = predict.call1((model, data))?.extract()?;
            Ok(pyarray.readonly().as_array().to_owned())
        })
    }
}
