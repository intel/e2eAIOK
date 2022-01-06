import torch
from warprnnt_pytorch import RNNTLoss
rnnt_loss = RNNTLoss()

# acts = torch.FloatTensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
#                             [0.1, 0.1, 0.6, 0.1, 0.1],
#                             [0.1, 0.1, 0.2, 0.8, 0.1]],
#                             [[0.1, 0.6, 0.1, 0.1, 0.1],
#                             [0.1, 0.1, 0.2, 0.1, 0.1],
#                             [0.7, 0.1, 0.2, 0.1, 0.1]]]])
# labels = torch.IntTensor([[1, 1]])
# act_length = torch.IntTensor([2])
# label_length = torch.IntTensor([2])

# large tensor size will cause segment fault
acts = torch.ones(32,790,309,256, requires_grad=True)
labels = torch.ones(32,308)
act_length = torch.ones(32) * 790
label_length = torch.ones(32) * 308
# fail with 128 bs
# acts = torch.ones(6, 264, 94, 1024, requires_grad=True)
# labels = torch.ones(6, 93)
# act_length = torch.ones(6) * 264
# label_length = torch.ones(6) * 93
print(f'acts: {acts.shape}, act len: {act_length.shape}, labels: {labels.shape}, label len: {label_length.shape}')

if acts.dtype != torch.float:
    acts = acts.float()
if labels.dtype != torch.int32:
    labels = labels.int()
if act_length.dtype != torch.int32:
    act_length = act_length.int()
if label_length.dtype != torch.int32:
    label_length = label_length.int()


# acts = torch.autograd.Variable(acts, requires_grad=True)
# labels = torch.autograd.Variable(labels)
# act_length = torch.autograd.Variable(act_length)
# label_length = torch.autograd.Variable(label_length)
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1, skip_first=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./results/trace'),
    with_modules=True) as prof:
    for i in range(10):
        print(f'step {i}')
        loss = rnnt_loss(acts, labels, act_length, label_length)
        loss.backward()
        prof.step()

prof.key_averages().table(sort_by='self_cpu_time_total')