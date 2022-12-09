import torch
from torch import nn
from pytorch3d.io import IO
from torch.utils.data import Dataset, DataLoader
from pytorch3d.ops import sample_points_from_meshes

def svd_projection(R: torch.Tensor) -> torch.Tensor:
    """
    Project R onto SO(3) using the SVD.

    Reference: https://proceedings.neurips.cc/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf
    """
    # TODO: test
    U, _, Vh = torch.linalg.svd(R)
    det = torch.linalg.det(torch.matmul(U, Vh))
    if len(R.shape) > 2:
        S_prime = torch.stack([torch.eye(3) for _ in range(R.shape[0])])
        S_prime[:, 2, 2] = det
    elif len(R.shape) == 2:
        S_prime = torch.eye(3)
        S_prime[2, 2] = det
    else:
        raise ValueError(f"Input tensor R (shape={R.shape}) has too few  dimensions.")

    return torch.matmul(torch.matmul(U, S_prime), Vh)

class RotationData(Dataset):
    def __init__(self,
        mesh_file_path: str = "../models/airplane.obj",
        num_points: int = 100,
        dataset_size: int = 2400,
        device: str = "cpu"
    ) -> None:
        mesh = IO().load_mesh(mesh_file_path, device=device)
        self.point_cloud = torch.flatten(sample_points_from_meshes(mesh, num_samples = num_points, return_normals = False, return_textures = False),0,1)
        self.dataset_size = dataset_size
        self.rotation_matrices = svd_projection(torch.rand(dataset_size,3,3))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return torch.matmul(self.point_cloud,self.rotation_matrices[idx]),self.rotation_matrices[idx]

class FFN_block(nn.Sequential):
    """
    Feed forward block
    """
    def __init__(self, emb_in: int, emb_out: int):
        super().__init__(
            nn.ReLU(),
            nn.Linear(emb_in, emb_out),
        )

class head (nn.Sequential):
    """
    A ReLU stack that takes in a tensor of (b x in_size) and outputs (b x outsize)
    """
    def __init__(self, in_size: int = 128, emb_size: int = 256, out_size: int = 6, depth: int = 3):
        super().__init__(*([nn.Linear(in_size, emb_size),]+[FFN_block(emb_size, emb_size) for _ in range(depth)]+[nn.Linear(emb_size, out_size)]))

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = torch.flatten(X,1,2).to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        pred = torch.reshape(pred, (pred.size()[0],3,3))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = torch.flatten(X,1,2).to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            pred = torch.reshape(pred, (pred.size()[0],3,3))
            loss = loss_fn(pred, y)

            test_loss += loss
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

device = "cpu"
model = model = head(300,256,9,3).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

temp = RotationData()

batchSize = 10
trainset, testset = torch.utils.data.random_split(temp,[2000,400])
train_loader = DataLoader(dataset=trainset, batch_size=batchSize,shuffle=True)
test_loader = DataLoader(dataset=testset,batch_size=batchSize,shuffle=True)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")




