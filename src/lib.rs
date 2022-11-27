mod pyvtable;
mod types;

use std::sync::Mutex;

use self::pyvtable::PyVTable;
use self::types::{Data, Labels, PredictedLabels, PyModel};

#[non_exhaustive]
pub struct Classifier {
    model: Mutex<PyModel>,
}

impl Classifier {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        Ok(Classifier {
            model: Mutex::new(PyVTable::load(path)?),
        })
    }

    pub fn new() -> anyhow::Result<Self> {
        Ok(Classifier {
            model: Mutex::new(PyVTable::define()?),
        })
    }

    pub fn predict(&self, data: Data) -> anyhow::Result<PredictedLabels> {
        let model = self.model.lock().unwrap();
        PyVTable::predict(&model, data)
    }

    pub fn train(&self, data: Data, labels: Labels) -> anyhow::Result<f32> {
        let model = self.model.lock().unwrap();
        PyVTable::train(&model, data, labels)
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let model = self.model.lock().unwrap();
        PyVTable::save(&model, path)
    }
}
