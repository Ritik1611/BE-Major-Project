#[derive(Clone)]
pub struct Round {
    pub id: u64,
    pub model_version: String,
    pub epsilon_max: f64,
    pub upload_uri: String,
}

impl Round {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            model_version: "v1".into(),
            epsilon_max: 1.0,
            upload_uri: format!("objectstore://round-{}", id),
        }
    }
}
