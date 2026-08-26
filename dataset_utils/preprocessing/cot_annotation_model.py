import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

class CoTAnnotationModel():
    def __init__(self, config: dict):
        super().__init__()
        model_path = config['pretrained_model_path']
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                        device_map="auto",
                        max_memory=config.get("max_memory")
                    )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def vlm_inference(self, inputs):
        inputs = self.processor(
            text=[inputs['text']],
            images=inputs['image_inputs'],
            videos=inputs['video_inputs'],
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)
        outputs = self.vlm.generate(**inputs, max_new_tokens=700)
        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        output_text = self.processor.batch_decode(
            outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text