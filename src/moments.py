from torch import nn

class Moments(nn.Module):

    # NOTA: la activación final va a depender del tipo de dato que tengamos
    # Como vamos a empezar por MNIST, directamente ponemos una sigmoide, 
    # más adelante habrá que añadir opciones para diferentes modelos de observación (logístico, gaussiano...)

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__() 

        # NN Layers
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        # Activations
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.hidden(x)
        out = self.activation(out)
        out = self.output(out)
        self.out = self.sigmoid(out)