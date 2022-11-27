use numpy::IntoPyArray;
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
    fn_predict: Py<PyAny>,
    fn_define: Py<PyAny>,
    fn_train: Py<PyAny>,
    fn_save: Py<PyAny>,
}

static SOURCE: &str = include_str!("source.py");

// pub fn init() -> anyhow::Result<&'static Classifier> {
//     static INSTANCE: OnceCell<Classifier> = OnceCell::new();
//     INSTANCE.get_or_try_init(|| {
//         let inner = Python::with_gil(|py| {
//             let source = PyModule::from_code(py, SOURCE, "source.py", "source")?;
//
//             let model = source.getattr("load_model")?.call1((MODEL_PATH,))?.into();
//             let predict = source.getattr("predict")?.into();
//             let define = source.getattr("define_model")?.into();
//             let train = source.getattr("train_model")?.into();
//             let save = source.getattr("save_model")?.into();
//             anyhow::Ok(Inner {
//                 model,
//                 fn_predict: predict,
//                 fn_define: define,
//                 fn_train: train,
//                 fn_save: save,
//             })
//         })?;
//         let inner = Mutex::new(inner);
//         Ok(Classifier { inner })
//     })
// }

impl Inner {
    fn from_file(path: &str) -> anyhow::Result<Self> {
        let inner = Python::with_gil(|py| {
            let source = PyModule::from_code(py, SOURCE, "source.py", "source")?;

            let model = source.getattr("load_model")?.call1((path,))?.into();

            let fn_predict = source.getattr("predict")?.into();
            let fn_define = source.getattr("define_model")?.into();
            let fn_train = source.getattr("train_model")?.into();
            let fn_save = source.getattr("save_model")?.into();

            anyhow::Ok(Inner {
                model,
                fn_predict,
                fn_define,
                fn_train,
                fn_save,
            })
        })?;
        Ok(inner)
    }

    fn predict(&self, data: ndarray::Array4<f64>) -> anyhow::Result<ndarray::Array2<f32>> {
        Python::with_gil(|py| {
            let data = data.into_pyarray(py);
            let model = self.model.as_ref(py);
            let predict = self.fn_predict.as_ref(py);
            let pyarray: &numpy::PyArray2<f32> = predict.call1((model, data))?.extract()?;
            Ok(pyarray.readonly().as_array().to_owned())
        })
    }
}

impl Classifier {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let inner = Mutex::new(Inner::from_file(path)?);
        Ok(Classifier { inner })
    }

    pub fn predict(&self, data: ndarray::Array4<f64>) -> anyhow::Result<ndarray::Array2<f32>> {
        self.inner.lock().unwrap().predict(data)
    }
}
