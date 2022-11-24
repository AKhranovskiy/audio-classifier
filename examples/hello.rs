fn main() {
    let classifier = audio_classifier::init();
    classifier.hello().expect("Python executed");
}
