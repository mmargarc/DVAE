import torch
from torchvision import datasets, transforms
from src.dvae import DVAE
import matplotlib.pyplot as plt

import wandb
#wandb.login()

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])                             

# Download and load the training data
batch_size = 128
trainset = datasets.MNIST('./data/MNIST/', download=False, train=True, transform=transform)
part_tr = torch.utils.data.random_split(trainset, [5000,len(trainset)-5000])[0] # just 5k images
trainloader = torch.utils.data.DataLoader(part_tr, batch_size=batch_size, shuffle=True)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)  # whole dataset

'''
# Download and load the test data
testset = datasets.MNIST('./data/MNIST/', download=False, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
'''

# Show an image
dataiter = iter(trainloader)
images, labels = dataiter.next()
plt.imshow(images[1].numpy().reshape([28,28]), cmap='Greys_r')

# DVAE parameters
D = 28*28
K = 128
hdim = 10
learning_rate = 1e-4
beta = 5
device = 0 # cuda device
n_epochs = 200

# Init W&B
wandb.init(
    project="pruebas-DVAE-MNIST",
    config={
        "learning_rate": learning_rate,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "beta": beta,
        "K": K,
        "hidden_dimension": hdim
        })

dvae = DVAE(D, K, hdim, beta=beta, lr=learning_rate, cuda_device=device)

ELBO = []
kl = []
reconstruction = []

for e in range(n_epochs):

    ELBO_epoch = 0
    kl_epoch = 0
    reconstruction_epoch = 0

    for images, labels in trainloader:    # Batches

        images_forward = images.view(images.shape[0], -1)

        dvae.sgd_step(images_forward)

        ELBO_epoch += dvae.ELBO_loss.cpu().data.numpy()
        kl_epoch += dvae.kl.cpu().data.numpy()
        reconstruction_epoch += dvae.reconstruction.cpu().data.numpy()

    ELBO.append(-ELBO_epoch/len(trainloader))
    kl.append(kl_epoch/len(trainloader))
    reconstruction.append(reconstruction_epoch/len(trainloader))

    # Log train metrics to W&B
    wandb.log({
                "ELBO": -ELBO_epoch/len(trainloader), 
                "KL": kl_epoch/len(trainloader), 
                "reconstruction": reconstruction_epoch/len(trainloader)
            })

    if(e % 10 == 0): # Every 10 epochs
        print("ELBO loss after %d epochs: %f" 
                %(e, ELBO[-1]))

# Close W&B run 
wandb.finish()

plt.figure()
plt.plot(ELBO, label='ELBO')
plt.title('ELBO evolution')
plt.show()

plt.figure()
plt.plot(kl, label='KL')
plt.title('KL term evolution')
plt.show()

plt.figure()
plt.plot(reconstruction, label='reconstruction')
plt.title('Reconstruction term evolution')
plt.show()

# Plot a generated image
dvae.encoder.eval()
dvae.decoder.eval()
dvae.sample_from_q_z()
dvae.decoder.forward(dvae.q_z)
for i in range(len(dvae.decoder.out.cpu().data.numpy())):
    plt.figure()
    plt.imshow(dvae.decoder.out.cpu().data.numpy()[i,:].reshape([28,28]), cmap='Greys_r')
