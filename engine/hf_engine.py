import time
from typing import Callable,Literal
import torch
from pydantic import BaseModel,Field,computed_field
from transformers import AutoTokenizer,AutoModelForCausalLM


FinishReason=Literal["stop","length"]
TokenCallback=Callable[[int,str,FinishReason|None],None]


class GenRequest(BaseModel):
    """One request as the engine sees it (validated)."""
    prompt:str=Field(min_length=1)
    max_tokens:int =Field(default=128,ge=1)
    temperature:float=Field(default=0.0,ge=0.0,le=2.0)
    ignore_eos:bool=False


class GenResult(BaseModel):
    """What one row produced."""
    text:str
    token_ids:list[int]
    finish_reason:FinishReason
    prompt_tokens:int


class BatchStats(BaseModel):
    """Waste ledger for one batch (2.1.4). Logged by Triton's model.py."""

    batch_size:int
    prompt_lens:list[int]
    output_lens:list[int]
    decode_steps:int
    prefill_s:float
    decode_s:float
    @computed_field
    @property
    def w_pre(self)->float:
        "prefill tokens that are pad"
        return 1-sum(self.prompt_lens)/(self.batch_size*max(self.prompt_lens))

    @computed_field
    @property
    def eta(self)->float:
        "usefull output slots"
        return sum(self.output_lens)/(self.batch_size*max(self.output_lens))


class HFStaticEngine:
    def __init__(self, model_id: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.device=device

        self.tokenizer=AutoTokenizer.from_pretrained(model_id,padding_side="left")
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=dtype).to(self.device)
        self.model.eval()
        eos=self.model.generation_config.eos_token_id
        eos_list = eos if isinstance(eos, (list, tuple)) else [eos]
        self.eos_ids: set[int] = {int(t) for t in eos_list if t is not None}
        
        self.last_stats:BatchStats|None=None


    def _sample(self, logits: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, V] float, temps: [B] float. Returns next token ids [B] (long).
        Greedy rows (temp == 0) take argmax; others sample from softmax(logits / temp).
        """
        logits=logits.float()
        greedy=logits.argmax(-1)
        safe_t=torch.where(temps==0,torch.ones_like(temps),temps)
        probs=torch.softmax(logits/safe_t[:,None],dim=-1)
        sampled=torch.multinomial(probs,num_samples=1).squeeze(-1)
        return torch.where(temps==0,greedy,sampled)
    @torch.inference_mode()
    def generate_batch(self, reqs: list[GenRequest],
                       on_token: TokenCallback | None = None) -> list[GenResult]:
        """
        Run one static batch to completion.
        """
        B=len(reqs)
        enc=self.tokenizer([r.prompt for r in reqs],return_tensor="pt",padding=True).to(self.device)
        input_ids,attn=enc["input_ids"],enc["attention_mask"]
        prompt_lens=attn.sum(dim=1).tolist()
        positions=(attn.cumsum(dim=-1)-1).clamp(min=0)
        temps=torch.tensor([r.temperature for r in reqs],dtype=torch.float32, device=self.device)
        budgets=[r.max_tokens for r in reqs]
        generated: list[list[int]]=[[]for _ in range(B)]
        done=[False]*B
        finish_reasons:list[FinishReason|None]=[None]*B


        torch.cuda.synchronize(self.device)
        prefill_start=time.perf_counter()
        out=self.model(input_ids=input_ids,attention_mask=attn,position_ids=positions,use_cache=True)
        past=out.past_key_values
        next_tok=self._sample(out.logits[:,-1,:],temps)
        torch.cuda.synchronize(self.device)
        prefill_s=time.perf_counter()-prefill_start

        decode_steps=0
        torch.cuda.synchronize(self.device)
        decode_start=time.perf_counter()
        while True:
            toks=next_tok.tolist()
            decode_steps+=1
            for i,token_value in enumerate(toks):
                if done[i]:
                    continue
                token_id=int(token_value)
                generated[i].append(token_id)
                hit_eos=(not reqs[i].ignore_eos and token_id in self.eos_ids)
                hit_len=len(generated[i])>=budgets[i]
                if hit_eos:
                    reason:FinishReason|None="stop"
                elif hit_len:
                    reason="length"
                else:
                    reason=None

                piece=("" if hit_eos else self.tokenizer.decode([token_id],skip_special_tokens=True))
                if reason is not None:
                    done[i]=True
                    finish_reasons[i]=reason
                if on_token is not None:
                    on_token(i,piece,reason)
            if all(done):
                break
            ones=torch.ones((B,1),dtype=attn.dtype,device=self.device)
            attn=torch.cat((attn,ones),dim=1)
            positions=positions[:,-1:]+1
            out=self.model(
                input_ids=next_tok[:,None],
                attention_mask=attn,
                position_ids=positions,
                past_key_values=past,
                use_cache=True
            )
            past=out.past_key_values
            next_tok=self._sample(out.logits[:,-1,:],temps)
        torch.cuda.synchronize(self.device)
        decode_s=time.perf_counter()-decode_start

        

                
        output_lens=[len(ids) for ids in generated]
        self.last_stats=BatchStats(
            batch_size=B,
            prompt_lens=prompt_lens,
            output_lens=output_lens,
            decode_steps=decode_steps,
            prefill_s=prefill_s,
            decode_s=decode_s,
        )
        return [
            GenResult(
                text=self.tokenizer.decode(generated[i], skip_special_tokens=True),
                token_ids=generated[i],
                finish_reason=finish_reasons[i],
                prompt_tokens=prompt_lens[i],
            )
            for i in range(B)
        ]



