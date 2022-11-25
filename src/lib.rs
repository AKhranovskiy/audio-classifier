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

    pub fn verify(&self) -> anyhow::Result<f64> {
        let f = std::fs::File::open("./data.pickle")?;
        let reader = std::io::BufReader::new(f);

        let data: Vec<Vec<f64>> = serde_pickle::from_reader(reader, Default::default())?;
        assert_eq!(3, data.len());

        let min_len = data.iter().map(Vec::len).min().unwrap();
        assert_eq!(0, min_len % (150 * 39));

        let data = data
            .into_iter()
            .flat_map(|v| v.into_iter().take(min_len))
            .collect::<Vec<_>>();

        let number_of_images = data.len() / (150 * 39);
        dbg!(&number_of_images);

        let data = ndarray::Array4::from_shape_vec((number_of_images, 150, 39, 1), data)?;
        dbg!(data.shape());

        let labels = ndarray::concatenate![
            ndarray::Axis(0),
            ndarray::Array1::from_elem((number_of_images / 3,), 0),
            ndarray::Array1::from_elem((number_of_images / 3,), 1),
            ndarray::Array1::from_elem((number_of_images / 3,), 2)
        ];

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
