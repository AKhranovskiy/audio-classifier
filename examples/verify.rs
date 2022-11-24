fn main() {
    let result = audio_classifier::init().verify().expect("Python function returned result");
    println!("Accuracy {:2.02}%", result * 100.0)
}
