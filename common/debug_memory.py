# prints currently alive Tensors and Variables
import torch
import gc

def print_active_tensors():
    cnt_cpu = 0
    cnt_gpu = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
            else:
                print('others', type(obj), obj.size())
            if obj.device.type == 'gpu':
                cnt_gpu += obj.numel() * 4
            else:
                cnt_cpu += obj.numel() * 4
        except Exception as e:
            #print(type(e))
            #print(e.args)
            #print(e)
            #import pdb;pdb.set_trace()
            pass
    print(f'total size on GPU: {cnt_gpu/1024/1024/1024} GB')
    print(f'total size on CPU: {cnt_cpu/1024/1024/1024} GB')
