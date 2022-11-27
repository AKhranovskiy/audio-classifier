use std::path::Path;

use audio_classifier::Classifier;
use ndarray::Axis;
use ndarray_stats::QuantileExt;

fn main() -> anyhow::Result<()> {
    let (data, labels) = prepare_data("./data.pickle")?;

    let classifier = Classifier::from_file("./model")?;

    // Parallel execution is not possible due to Python GIL.
    let owned_results = data
        .axis_chunks_iter(Axis(0), 413)
        .map(|chunk| {
            classifier
                .predict(chunk.to_owned())
                .expect("Python function returned result")
        })
        .collect::<Vec<_>>();

    let result = ndarray::concatenate(
        Axis(0),
        owned_results
            .iter()
            .map(|v| v.view())
            .collect::<Vec<_>>()
            .as_ref(),
    )?;

    let predicted = result
        .rows()
        .into_iter()
        .map(|a| a.argmax())
        .collect::<Result<Vec<usize>, _>>()?;

    let correct = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(&p, &c)| p as u32 == c)
        .count();

    println!(
        "Accuracy {:2.02}%",
        (correct as f32) / (labels.len() as f32) * 100.0
    );

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
