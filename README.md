# Animorphs

## __Project Outline__ 
Our goal is to make a model architecture that can morph one object into another. We input two images (start and goal image) and output and the main task of our model is to use their latent space vectors of the two images to come up with intermediary latent vectors. We then use the latent vectors to generate intermediate images. Finally, we will string together the start, intermediate images, and goal image to make a smooth .gif-like animation.

## __Model Architecture__ 

### __Human Faces Dataset__
- For human faces, we take advantage of the pre-trained styleGAN model built by NVIDIA. Since we can generate an image from any vector now, the image generation part was taken care of. Now we had to encode our start and goal images to obtain the intermediary latent vectors. We try to approximate using two methods.
#### __Latent Vector Approximation using MSE__
- This is our first naive approach where the steps are described below.
  1. Initialize a random vector, $w$, of dimension $[1, 18, 512]$ and minimal loss vector w_min of same dimension to zero (only in the first iteration)
  2. Pass in the vector, $w$, to generator $G$ of styleGAN model.
  3. Compare the image $I_{w}$, the original image, and the generated image, $I_{G(w)}$, using MSE loss.
  4. If the loss is the smallest we have seen so far, do the update, $w_{min} = w$ and go back to step 1.
   
- With this method we were able to achieve the results below with five input images.
![](./data/naive1.png)
(0 maps to 3, 1 -> 0, 2 -> 4, 3 -> 1, 4 -> 2)

#### __Latent Vector Approximation using e4e__
- This new technique, which was published here, is a more generalize version where if you pass in an image, you can immediately get its latent vector (Note that our first approach requires us to compute one image at a time so it is very inefficient hence naive). The model structure is as follows:
![](./data/model_arch.png) 
- The steps are:
  - You pass in an image to the encoder, $E$, and it outputs a latent vector, $w$, and $N$ offsets (for face $N=18$), $\delta$. We combine these two 
  and pass it into the generator G of pretrained styleGAN model.
  - We again compute the MSE loss of the original image and generated image. However, at the same time, we also set the loss for our offsets delta and we try to minimize these two losses.
  - We update the learnable parameters in the encoder, $E$.
  
- With this method the following results were achieved:
![](./data/e4e.png)  

## Morphing 
- To be able to morph from one image to another, we researched ways to combine latent vectors. We decided to use linear interpolation to calculate intermediary vectors between two image coordinates. The formula for `linear interpolation` is defined as such:
$$\begin{align}
w_i = \alpha w_1 + (1-\alpha) w_2
\end{align}$$
- where $\alpha$ is the interpolation factor and $w_i$ is the interpolation vector. ($w_1$ and $w_2$ are the two latent vectors, one for start image and the other for goal image.) By running the generator on several $\alpha$ values, we can get same number of intermediary images, and by connecting them, obtain a smooth animation. 

### Human Face Morphing
- Using the 5 approximation images obtained in the naive method for Human Face, the following morphing was achieved:
  
<p align='center'>
<img src = './data/Kenta_in_styleGAN.gif'>
</p>

- Additionally, with encoder4editor, we were now able to explore the latent space of the human faces dataset and 'edit' our image in certain directions. For our project, we have made it so that we can edit `age`, `pose`, and `age + pose` attributes.

Moving in `age` direction:
<p align='center'>
<img src = './data/kb_age.gif'
width=250>
</p>

Moving in `pose` direction:
<p align='center'>
<img src = './data/naveen.gif'
width=250>
</p>

Moving in `age + pose` direction:
<p align='center'>
<img src = './data/kenta_age+pos.gif'
width=250>
</p>
