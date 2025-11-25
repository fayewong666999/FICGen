import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()
    
    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        
    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]
    

class HOIPositionNetV5(nn.Module):
    def __init__(self, out_dim=768, fourier_freqs=8, max_objects=15):
        super().__init__()
        self.out_dim = out_dim
        
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        
        self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)
        self.position_dim = fourier_freqs * 2 * 8
        
        self.linear_entity = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.linear_interaction = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim)
        )
        
        self.saf = nn.Linear(self.out_dim*3, self.out_dim)
        
    def polygon_to_box(self, polygon):
        x_coords = polygon[0::2]
        y_coords = polygon[1::2]
        
        return torch.tensor([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()],
                           dtype = polygon.dtype, device = polygon.device)
    
    def get_intersection_polygon(self, box1, box2):
        
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        if x2<=x1 or y2<=y1:
            return None
        return torch.tensor([x1, y1, x2, y1, x2, y2, x1, y2],dtype=box1.dtype, device=box1.device)
    
    def forward(self, polygons):
        N = polygons.shape[0]
        device = polygons.device
        output = []
        for i in range(N):
            tokens = []
            subj_poly = polygons[i]
            subj_box = self.polygon_to_box(subj_poly)
            subj_pos = self.fourier_embedder(subj_poly.unsqueeze(0)).squeeze(0)
            has_overlap = False
            for j in range(N):
                if i == j:
                    continue
                obj_poly = polygons[j]
                obj_box = self.polygon_to_box(obj_poly)
                inter_poly = self.get_intersection_polygon(subj_box, obj_box)
                
                if inter_poly is None:
                    continue
                has_overlap = True
                obj_pos = self.fourier_embedder(obj_poly.unsqueeze(0)).squeeze(0)
                
                inter_pos = self.fourier_embedder(inter_poly.unsqueeze(0)).squeeze(0)
                
                subj_token = self.linear_entity(subj_pos)
                inter_token = self.linear_interaction(inter_pos)
                obj_token = self.linear_entity(obj_pos)
                
                subj_token = subj_token + self.position_embedding.emb(torch.tensor(0).to(subj_token.device))
                inter_token = inter_token + self.position_embedding.emb(torch.tensor(1).to(inter_token.device))
                obj_token = obj_token + self.position_embedding.emb(torch.tensor(2).to(obj_token.device))
                
                tokens.append(self.saf(torch.cat([subj_token, inter_token, obj_token], dim=-1)).unsqueeze(0))
                
            if not has_overlap:
                subj_token = self.linear_entity(subj_pos)
                inter_token = self.linear_interaction(inter_pos)
                obj_token = self.linear_entity(obj_pos)
           
                tokens.append(self.saf(torch.cat([subj_token, inter_token, obj_token], dim=-1)).unsqueeze(0))
           
            tokens = torch.cat(tokens,dim=0)
            output.append(tokens.mean(0))
            
                
        out = torch.stack(output, dim=0)
        
        return out
    
if __name__ == "__main__":
    polygons = torch.tensor([[0.3475, 0.3550, 0.3550, 0.3475, 0.5411, 0.5411, 0.5767, 0.5767],
        [0.3544, 0.4500, 0.4500, 0.3544, 0.5367, 0.5367, 0.6033, 0.6033],
        [0.5562, 0.5794, 0.5794, 0.5562, 0.5111, 0.5111, 0.5378, 0.5378],
        [0.0731, 0.0856, 0.0856, 0.0731, 0.5411, 0.5411, 0.5867, 0.5867],
        [0.5713, 0.5938, 0.5938, 0.5713, 0.5067, 0.5067, 0.5344, 0.5344],
        [0.5869, 0.6050, 0.6050, 0.5869, 0.5078, 0.5078, 0.5311, 0.5311],
        [0.1694, 0.1794, 0.1794, 0.1694, 0.5356, 0.5356, 0.5967, 0.5967],
        [0.2169, 0.2256, 0.2256, 0.2169, 0.5378, 0.5378, 0.5889, 0.5889],
        [0.5431, 0.5688, 0.5688, 0.5431, 0.5156, 0.5156, 0.5411, 0.5411],
        [0.2725, 0.2825, 0.2825, 0.2725, 0.5356, 0.5356, 0.5878, 0.5878],
        [0.5806, 0.6481, 0.6481, 0.5806, 0.5311, 0.5311, 0.6322, 0.6322],
        [0.1875, 0.1950, 0.1950, 0.1875, 0.5433, 0.5433, 0.5911, 0.5911],
        [0.2006, 0.2100, 0.2100, 0.2006, 0.5367, 0.5367, 0.5856, 0.5856],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
    )
    hoi = HOIPositionNetV5()
    print(hoi(polygons).shape)
    
    