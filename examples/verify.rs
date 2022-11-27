use std::path::Path;

fn main() -> anyhow::Result<()> {
    let (data, labels) = prepare_data("./data.pickle")?;

    let result = audio_classifier::init()?
        .lock()
        .unwrap()
        .verify(data, labels)
        .expect("Python function returned result");

    println!("Accuracy {:2.02}%", result * 100.0);

    Ok(())
}

fn prepare_data<P>(source: P) -> anyhow::Result<(ndarray::Array4<f64>, ndarray::Array1<u32>)>
where
    P: AsRef<Path>,
{
    let f = std::fs::File::open(source)?;
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

    Ok((data, labels))
}
