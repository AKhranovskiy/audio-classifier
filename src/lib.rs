use pyo3::PyResult;

#[non_exhaustive]
pub struct Classifier;

static CLASSIFIER: Classifier = Classifier {};

pub fn init() -> &'static Classifier {
    &CLASSIFIER
}

impl Classifier {
    pub fn hello(&self) -> PyResult<()> {
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
}
