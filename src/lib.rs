#[non_exhaustive]
pub struct Classifier;

static CLASSIFIER: Classifier = Classifier {};

pub fn init() -> &'static Classifier {
    &CLASSIFIER
}

// SHAPE 150x39

impl Classifier {
    pub fn hello(&self) -> anyhow::Result<()> {
        use pyo3::prelude::*;
        use pyo3::types::IntoPyDict;

        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let version: String = sys.getattr("version")?.extract()?;

            let locals = [("os", py.import("os")?)].into_py_dict(py);
            let code = r#"os.getenv('USER') or os.getenv('USERNAME') or 'Unknown'"#;
            let user: String = py.eval(code, None, Some(locals))?.extract()?;

            println!("Hello {}, I'm Python {}", user, version);
            Ok(())
        })
    }

    pub fn train_batch(&self) -> anyhow::Result<()> {
        todo!()
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
