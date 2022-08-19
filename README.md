# SlimBERT - BERT Compression with Neural Slimming


We study how to compress the BERT model in a structured pruning
manner. We proposed the neural slimming technique to assess the importance of each
neuron and designed the cost function and pruning strategy to remove neurons that make
zero or less contribution to the prediction. After getting fine-tuned on the downstream
tasks, the model can learn a more compact structure, and we name it SlimBERT.

## Methods

To estimate the contribution of each neuron, we introduce a importance factor α which
is a learnable parameter. A slim layer is a group
of independent importance factors that are individually optimized. Each time we want to
perform pruning on a certain layer, we connect the slim layer to this layer.

### Slim Layer And Loss Function
<img width="903" alt="image" src="https://user-images.githubusercontent.com/25696555/185279658-4da1838c-b917-4130-8766-bfb8dbc2ecd0.png">

Due to the flexibility of the slim layer, we can easily apply it to the parts that we want
to prune in the model. For BERT, we use it on all the layers, including the embedding
layer, the multi-head self-attention layers, and the fully connected layer.
### SilmBERT Architecture
<img width="485" alt="image" src="https://user-images.githubusercontent.com/25696555/185279985-c4e4a8d8-753d-4c06-b0bb-f3eec286733e.png">


## Results
We tested our method on 7 GLUE tasks and used only 10% of the original parameters to
recover 94% of the original accuracy. It also reduced the run-time memory and increased the
inference speed at the same time. Compared to knowledge distillation methods and other
structured pruning methods, the proposed approach achieved better performance under
different metrics with the same compression ratio. Moreover, our method also improved
the interpretability of BERT. By analyzing neurons with a significant contribution, we can
observe that BERT utilizes different components and subnetworks according to different
tasks.

### Performance on GLUE Tasks
<img width="712" alt="image" src="https://user-images.githubusercontent.com/25696555/185279397-069cbe03-05c3-40b8-8e64-fb9a22644046.png">

### Performance of SlimBERT vs Unstructured Pruning
<img width="516" alt="image" src="https://user-images.githubusercontent.com/25696555/185279540-d749c8b9-d8b6-4de9-9ce1-8a4b215671ca.png">

### Percentage of Remaining Neurons by Layers
<img width="351" alt="image" src="https://user-images.githubusercontent.com/25696555/185281326-bface675-5603-469f-837c-402a363fd70e.png">

### Percentage of removed neurons in each sublayer
<img width="513" alt="image" src="https://user-images.githubusercontent.com/25696555/185281406-c795b614-2b17-462f-980f-7d0c6a25d972.png">

### Important Attention Heads in SlimBERT
Green: High importance    Pink: Low importance
<img width="770" alt="image" src="https://user-images.githubusercontent.com/25696555/185281563-c8be4de2-f75b-43a1-b723-f0c8de2fbdcc.png">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/25696555/185281709-f82f9989-3c48-4518-b32c-38671628161a.png">

