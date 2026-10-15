import json
import numpy as np
import triton_python_backend_utils as pb_utils
from pydantic import ValidationError
from engine.hf_engine import HFStaticEngine,GenRequest
import torch

def _read_request(request) -> GenRequest:
    """Triton request -> validated GenRequest. Raises ValidationError on bad input."""
    tensor=pb_utils.get_input_tensor_by_name(
        request,"text_input"
    )
    prompt=tensor.as_numpy().reshape(-1)[0].decode("utf-8")
    params=json.loads(request.parameters() or "{}")
    return GenRequest.model_validate({"prompt":prompt,**params})

def _text_response(text: str):
    """One streamed piece -> pb_utils.InferenceResponse with a [1, 1] string tensor."""
    out=pb_utils.Tensor("text_output", np.array([[text.encode("utf-8")]], dtype=object))
    return pb_utils.InferenceResponse(output_tensors=[out])
def _error_response(message: str):
    """InferenceResponse carrying a TritonError."""
    return pb_utils.InferenceResponse(error=pb_utils.TritonError(message))


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        cfg=json.loads(args["model_config"])
        params=cfg["parameters"]
        model_id=params["model_id"]["string_value"]
        dtype_name=params["dtype"]["string_value"]
        device=f"cuda:{args['model_instance_device_id']}"
        dtype=getattr(torch,dtype_name)
        self.engine=HFStaticEngine(
            model_id=model_id,
            device=device,
            dtype=dtype
        )
        self.log=pb_utils.Logger
        self.log.log_info(f"Loaded {model_id} on {device} with {dtype_name}")


    def execute(self, requests: list) -> None:
 
        senders=[r.get_response_sender() for r in requests]
        finished=[False]*len(requests)

        valid_reqs=[]
        rows=[]

        for i,request in enumerate(requests):
            try:
                req=_read_request(request)
            except Exception as e:
                senders[i].send(
                    _error_response(str(e)),flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
                finished[i]=True
                continue
            valid_reqs.append(req)
            rows.append(i)

        if not valid_reqs:
            return None

        

        def on_token(row,piece,finish_reason):
            sender=senders[rows[row]]
            if sender.is_cancelled():
                if not finished[rows[row]]:
                    sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                    finished[rows[row]]=True
                return
            flags = (
                pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                if finish_reason is not None
                else 0
            )
            sender.send(_text_response(piece),flags=flags)
            if finish_reason is not None:
                finished[rows[row]]=True

        try:
            self.engine.generate_batch(valid_reqs,on_token=on_token)
        except Exception as e:
            self.log.log_error(f"HFStaticEngine failed: {e}")
            for i,sender in enumerate(senders):
                if finished[i]:
                    continue

                if sender.is_cancelled():
                    sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    sender.send(
                        _error_response(str(e)),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )

                finished[i]=True

  
        if self.engine.last_stats is not None:
            stats=self.engine.last_stats
            self.log.log_info(
                f"batch_size={stats.batch_size}, "
                f"w_pre={stats.w_pre:.3f}, "
                f"eta={stats.eta:.3f}, "
                f"decode_steps={stats.decode_steps}"
            )
        return None

    def finalize(self) -> None:
        self.log.log_info("Unloading HFStaticEngine")