use lazy_static::lazy_static;
use numpy::IntoPyArray;
use std::sync::Mutex;

use pyo3::types::PyModule;
use pyo3::{Py, PyAny, Python};

#[non_exhaustive]
pub struct Classifier {
    model: Mutex<PyModel>,
}

type PyModel = Py<PyAny>;

#[allow(dead_code)]
struct PyVTable {
    define: Py<PyAny>,
    load: Py<PyAny>,
    predict: Py<PyAny>,
    save: Py<PyAny>,
    train: Py<PyAny>,
}

impl PyVTable {
    fn init() -> Self {
        static SOURCE: &str = include_str!("source.py");

        Python::with_gil(|py| {
            let source = PyModule::from_code(py, SOURCE, "source.py", "source")
                .expect("Python source is loaded");

            let attr = |name: &str| source.getattr(name).expect("Attribute {name} is loaded");

            PyVTable {
                define: attr("define_model").into(),
                load: attr("load_model").into(),
                predict: attr("predict").into(),
                save: attr("save_model").into(),
                train: attr("train_model").into(),
            }
        })
    }
}

lazy_static! {
    static ref PYVTABLE: PyVTable = PyVTable::init();
}

pub type Data = ndarray::Array4<f64>;
pub type PredictedLabels = ndarray::Array2<f32>;
pub type Labels = ndarray::Array1<u32>;

impl Classifier {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let model =
            Python::with_gil(|py| anyhow::Ok(PYVTABLE.load.as_ref(py).call1((path,))?.into()))?;
        Ok(Classifier {
            model: Mutex::new(model),
        })
    }

    pub fn new() -> anyhow::Result<Self> {
        let model = Python::with_gil(|py| anyhow::Ok(PYVTABLE.define.as_ref(py).call0()?.into()))?;

        Ok(Classifier {
            model: Mutex::new(model),
        })
    }

    pub fn predict(&self, data: Data) -> anyhow::Result<PredictedLabels> {
        let model = self.model.lock().unwrap();
        Python::with_gil(|py| {
            let data = data.into_pyarray(py);
            let model = model.as_ref(py);
            let pyarray: &numpy::PyArray2<f32> = PYVTABLE
                .predict
                .as_ref(py)
                .call1((model, data))?
                .extract()?;
            Ok(pyarray.readonly().as_array().to_owned())
        })
    }

    pub fn train(&self, _data: Data, _labels: Labels) -> anyhow::Result<f32> {
        todo!()
    }

    pub fn save(&self, _path: &str) -> anyhow::Result<()> {
        todo!()
    }
}
