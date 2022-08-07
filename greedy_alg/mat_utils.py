import torch
import math

def parts_of_data_on_gpu(mat):
    # load a piece of data of at most 2GB to GPU
    max_size = 2 * 1024 * 1024 * 1024 
    element_size = 4
    num_parts = max(1, math.ceil(mat.numel() * element_size / max_size))
    return num_parts

def g_prod_gw_efficient(g, gw, num_parts=None):
    '''calculate: (g * gw[:,None]).mean(dim=0)
        inputs: 
            g: N x D
            gw:N
        output: D
    '''
    N,D = g.shape
    rst = torch.zeros(D).to(gw.device)
    if num_parts is None:
        num_parts = parts_of_data_on_gpu(g)
    piece = int(math.ceil(N / num_parts))
    for i in range(num_parts):
        idx_s = i * piece
        idx_e = min((i + 1) * piece, N)
        g_    = g[idx_s:idx_e,:].to(gw.device) 
        rst += (g_ * gw[idx_s:idx_e,None]).sum(dim=0)
        del g_
        torch.cuda.empty_cache()
    rst = rst / N
    return rst
    

def product_efficient_v1(mat, vec, num_parts=None):
    D = mat.shape[0]
    if num_parts is None:
        num_parts = parts_of_data_on_gpu(mat)
    piece = int(math.ceil(D / num_parts))
    rst   = torch.zeros(D).to(vec.device)
    for i in range(num_parts):
        try:
            idx_s = i * piece
            idx_e = min((i + 1) * piece, D) 
            rst[idx_s:idx_e] = mat[idx_s:idx_e,:].to(vec.device) @ vec
        except:
            import pdb;pdb.set_trace()
    return rst 

#def _add_outer_products_efficient_v1(mat, vec, num_parts=2):
#    piece = int(math.ceil(len(vec) / num_parts))
#    vec_len = len(vec)
#    for i in range(num_parts):
#        for j in range(num_parts):
#            mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
#                torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)],
#                          vec[j * piece:min((j + 1) * piece, vec_len)])
#            )
#
#def efficient_ger(v1, v2):
#	"""
#		v1: N x d1
#		v2: N x d2
#		return: d1 x d2 
#	"""
#    N,d1 = v1.shape
#    N,d2 = v2.shape
#    # To allow 2GB is used at most
#    bz      = math.ceil(2e9 / (d1*d2*8))
#    K     = math.ceil(N/bz)
#    print(f'K={K}, bz={bz}')
#    rst    = 0
#    for kk in range(K):
#        b,e = kk * bz, min(N, (kk+1)*bz)
#        rst += (mat1[b:e] @ mat2[b:e]).sum(dim=0)
#    rst = rst/N
#    return rst


if __name__ == "__main__":
    g = torch.randn(1000, 1000000)
    gw = torch.randn(1000).cuda()
    gg = g.cuda()
    rst1 = (gg * gw[:,None]).mean(dim = 0)
    del gg
    torch.cuda.empty_cache()
    rst2 = g_prod_gw_efficient(g, gw)
    import pdb;pdb.set_trace()
