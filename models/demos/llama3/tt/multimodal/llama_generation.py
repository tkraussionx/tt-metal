from models.demos.llama3.tt.multimodal.llama_vision_model import LlamaVisionModel

import importlib

llama_reference_model = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
llama_reference_image_transforms = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.image_transform"
)


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args, paged_attention_config=None, vllm=False):
        self.mesh_device = tt_args.mesh_device
        self.tt_model = CrossAttentionTransformer(
            model_args,
            self.mesh_device,
            state_dict,
            tt_args.cache_path,
            model_args.dtype,
            configuration,
        )

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits, device=self.mesh_device, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=3)
        )
        return logits[..., : self.params.vocab_size].float()

    @classmethod
    def initialize_vllm_model(cls, hf_config, t3k_mesh_device, max_batch_size):
        tt_model_args = TtModelArgs(t3k_mesh_device)
        checkpoint = torch.load(tt_model_args.consolidated_weights_path, map_location="cpu", weights_only=True)
        # TODO: get model_args
        model_args = None
        return cls(
            configuration=tt_model_args, state_dict=checkpoint, model_args=model_args, tt_args=tt_args, vllm=True
        )

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        multi_modal_kwargs=None,
    ):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache, multi_modal_kwargs=multi_modal_kwargs
            )
        else:
            return NotImplementedError("TtLlamaModelForGeneration.forward() only supports single token generation")

    def decode_forward(
        self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, multi_modal_kwargs=None
    ):
        batch = tokens.shape[0]

        # Get inputs on device
        tt_inp_emb, start_pos, rot_mat, cache_idxs_tt, tt_page_table = self.tt_model.prepare_device_inputs(
            tokens, start_pos, mode="decode", page_table=page_table
        )
        xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = (
            multi_modal_kwargs["xattn_caches"],
            multi_modal_kwargs["cross_attention_masks"],
            multi_modal_kwargs["full_text_row_masked_out_mask"],
        )

        tt_logits = self.tt_model(
            cache_idxs_tt, tt_inp_emb, xattn_caches, cross_attention_masks, full_text_row_masked_out_mask
        )

        logits = self._process_logits(tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users
        # del tt_logits

        return logits
