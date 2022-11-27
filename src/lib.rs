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

#[allow(dead_code)]
struct Inner {
    model: Py<PyAny>,
    predict: Py<PyAny>,
    define: Py<PyAny>,
    train: Py<PyAny>,
    save: Py<PyAny>,
}

static SOURCE: &str = include_str!("source.py");
static MODEL_PATH: &str = "./model";

pub fn init() -> anyhow::Result<&'static Classifier> {
    static INSTANCE: OnceCell<Classifier> = OnceCell::new();
    INSTANCE.get_or_try_init(|| {
        let inner = Python::with_gil(|py| {
            let source = PyModule::from_code(py, SOURCE, "source.py", "source")?;

            let model = source.getattr("load_model")?.call1((MODEL_PATH,))?.into();
            let predict = source.getattr("predict")?.into();
            let define = source.getattr("define_model")?.into();
            let train = source.getattr("train_model")?.into();
            let save = source.getattr("save_model")?.into();
            anyhow::Ok(Inner {
                model,
                predict,
                define,
                train,
                save,
            })
        })?;
        let inner = Mutex::new(inner);
        Ok(Classifier { inner })
    })
}

impl Classifier {
    pub fn predict(&self, data: ndarray::Array4<f64>) -> anyhow::Result<ndarray::Array2<f32>> {
        let inner = self.inner.lock().unwrap();
        let Inner { model, predict, .. } = &*inner;

        Python::with_gil(|py| {
            let data = data.into_pyarray(py);
            let model = model.as_ref(py);
            let predict = predict.as_ref(py);
            let pyarray: &numpy::PyArray2<f32> = predict.call1((model, data))?.extract()?;
            Ok(pyarray.readonly().as_array().to_owned())
        })
    }
}
