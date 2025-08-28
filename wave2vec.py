from transformers import Wav2Vec2Processor, Wav2Vec2Model
name = "facebook/wav2vec2-base-960h"
Wav2Vec2Processor.from_pretrained(name)
Wav2Vec2Model.from_pretrained(name)
