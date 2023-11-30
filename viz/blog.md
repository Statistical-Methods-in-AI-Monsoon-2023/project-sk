# Understanding Neural Network Loss Landscapes

In our exploration of neural network loss landscapes, we investigated how:

1. The convexity of loss landscapes correlates with trainability.
2. The sharpness/flatness of minimizers influences generalization.
3. Architectural choices impact the landscape and, consequently, generalization:
   1. Importance of skip connections.
   2. Influence of the number of filters.
   3. Impact of the number of layers.

### Key Contributions:
1. Identified that simple visualizations inadequately represent the sharpness/flatness of minimizers.
2. Introduced "filter normalization" to facilitate meaningful comparisons between different minimizers.
3. Uncovered the relationship between chaotic loss landscapes, reduced generalization error, and diminished trainability, especially in deep networks. Proposed solution:
   - Leveraging skip connections to promote flat minimizers and prevent chaotic behavior.
4. Quantitatively validated that 2D loss landscapes capture the majority of the geometry by examining the most negative eigenvalues of the Hessian around local minima, visualized through a heatmap.
5. Quantitatively demonstrated that optimization trajectories inhabit a low-dimensional space.


## Visualizing Neural Network Loss Functions
Neural networks are trained on feature vectors and labels by minimizing a loss function, L(θ), where θ represents the parameters of the network. The loss function evaluates how well the network predicts labels for data samples.

$$ L(\theta) = \frac{1}{m}\sum^m_{i=1}l(x_i,y_i;\theta) $$

### Curse of Dimensionality
Neural nets have thousands of parameters, leading to high-dimensional loss functions. Visualizing them is challenging, often restricted to 1D or 2D plots. Various methods aim to bridge this dimensionality gap.

### 1-Dimensional Linear Interpolation
Method: Use two parameter sets, θ and θ₀, and interpolate along the line connecting them.
Parameterization: A scalar parameter α defines the weighted average θ(α) = (1 − α)θ + αθ₀.
Visualization: Plot the function f(α) = L(θ(α)).
Applications: Commonly used to study sharpness and flatness of minima, and their dependence on factors like batch size.
Weaknesses of 1D Interpolation
Limited for Non-Convexities: Difficulty in visualizing non-convexities, hindering the understanding of local minima.
Neglects Batch Normalization and Symmetries: Fails to consider batch normalization or invariance symmetries in the network, potentially leading to misleading sharpness comparisons.

# Exploring Loss Surfaces with Contour Plots and Random Directions

To employ this approach, a center point θ∗ is chosen in the graph, along with two direction vectors, δ and η. The method involves plotting a function of the form:

1. 1D (Line) Case:
$$ f(α) = L(θ∗ + αδ) $$

2. 2D (Surface) Case:
$$ f(α, β) = L(θ∗ + αδ + βη) $$

This technique, explores trajectories of different minimization methods and demonstrates that distinct optimization algorithms locate different local minima within the 2D projected space.

- Applied to analyze trajectories of various minimization methods.
- Used to showcase differences in local minima found by different optimization algorithms.

However, due to the computational burden of 2D plotting, these methods often yield low-resolution plots of small regions, lacking a comprehensive representation of the complex non-convexity of loss surfaces.

## High-Resolution Visualizations
In this context, high-resolution visualizations are employed over large slices of weight space to gain a more nuanced understanding of how network design influences the non-convex structure of loss surfaces. But these require a lot of compute.


# Enhancing Visualization with Filter-Wise Normalization

This study relies on plots using random direction vectors, δ and η, sampled from Gaussian distributions. However, the inherent scale invariance of neural networks poses challenges in meaningful comparisons between different minimizers or networks.

## Scale Invariance Challenge
- Network Weight Invariance: Multiplying or dividing weights by a factor maintains network behavior due to scale invariance, especially with ReLU non-linearities and batch normalization.
- Implications: Scale differences can lead to deceptive interpretations of loss function behavior, affecting sharpness perception.

## Addressing Scale Invariance with Filter-Wise Normalization
- Objective: Eliminate scaling effects for meaningful plot comparisons.
- Process: Normalize random Gaussian direction vectors filter-wise, ensuring each filter's norm matches that of the corresponding filter in the original parameters.
- Application: Not restricted to convolutional layers; extends to fully connected layers, treating them as a 1 × 1 convolutional layer.

## Comparing Filter-Wise Normalization
- Affirmative Correlation: Filter-wise normalized plots show a correlation between sharpness and generalization error.
- Misleading Plots: Plots without filter normalization can be misleading in understanding loss surface characteristics.
- Superiority Over Layer-Wise Normalization: Filter-wise normalization demonstrates superior correlation between sharpness and generalization compared to layer-wise or no normalization.

# Navigating the Sharp vs Flat Dilemma in Minimizers

## Introduction of Filter Normalization
Section 4 introduces filter normalization as a pivotal concept, offering an intuitive rationale for its application. The focus shifts to investigating whether sharp minimizers exhibit better generalization compared to flat minimizers, emphasizing the correlation between sharpness and generalization error when filter normalization is employed.

## Sharpness-Guided Comparison
- Filter Normalization Impact: Enables meaningful side-by-side comparisons of sharpness.
- Distorted Non-Normalized Plots: Without filter normalization, sharpness in plots may appear distorted and unpredictable.
- Weight Norm Growth: Illustrates the steady growth of weight norms during training without constraints, emphasizing the role of weight decay.

## Filter-Normalized Comparisons
- Improved Sharpness Correlation: Filter-normalized plots showcase better correlation between sharpness and generalization error.
- Visualizing Minimizer Characteristics: Two random directions and contour plots reveal wider contours for small-batch minimizers with non-zero weight decay.

The study emphasizes the importance of filter normalization in accurate comparisons, challenging previous misleading sharpness interpretations. Sharpness, when considered in the context of filter normalization, aligns more closely with generalization error. Large batches may produce visually sharper minima, yet filter-normalized comparisons unveil nuanced distinctions with higher test error.

# Insights into Trainability: Non-Convexity Structure of Loss Surfaces

This section explores the trainability of neural networks by investigating the (non)convexity structure of loss surfaces. The study addresses the variations in the ease of minimizing neural loss functions, influenced by factors such as network architecture, skip connections, and initialization strategies.

## Trainability Observations
- Not all neural architectures are equally trainable. Skip connections play a vital role in training extremely deep networks.
- Trainability is highly dependent on the initial parameters from which training starts.

## Empirical Study Questions
- Investigate if loss functions exhibit significant non-convexity.
- Explore why non-convexity is problematic in some situations but not in others.
- Understand why some architectures are easier to train.
- Examine the sensitivity of results to the choice of initialization.

## Architectural Variances
### Network Depth Influence
- VGG-like networks (ResNet-20/56/110-noshort) exhibit a transition from nearly convex to chaotic behavior as depth increases.
- Shortcut connections prevent the transition to chaotic behavior, preserving consistent geometry across different depths.
  
### Wide Models vs Thin Models
- Wide-ResNets with increased filter numbers demonstrate loss landscapes without chaotic behavior, emphasizing the role of width in preventing chaos.
- Notable correlation between sharpness, network width, and test error.

### Implications for Network Initialization
- Loss landscapes appear partitioned into well-defined regions of low loss and convex contours, surrounded by high-loss chaotic regions.
- The partitioning of chaotic and convex regions explains the importance of good initialization strategies and the ease of training for "good" architectures.

## Landscape Geometry and Generalization
- Visually flatter minimizers consistently correspond to lower test error, reinforcing the importance of filter normalization for visualizing loss function geometry.
- Deep networks without skip connections (chaotic landscapes) result in worse training and test error.
- More convex landscapes, such as Wide-ResNets, generalize better with lower error values.

### Cautionary Note on Convexity Interpretation
- Viewing loss surfaces involves significant dimensionality reduction.
- Use of principle curvatures to measure convexity and identify dominant positive curvatures in low-dimensional surfaces.

### Confirmation through Eigenvalues Analysis
- Mapping the ratio $$|λmin /λmax|$$ across the loss surfaces confirms that convex-looking regions correspond to areas with insignificant negative eigenvalues.
- Visualization captures significant non-convex features, reassuringly aligning with eigenvalues analysis.

The study provides valuable insights into neural network trainability, emphasizing the role of architecture, skip connections, and initialization strategies in shaping the (non)convexity structure of loss surfaces.

# Visualizing Optimization Paths: From Random to PCA Directions

We explore some methods for effectively visualizing optimization trajectories, exploring the limitations of random directions and proposing an approach based on Principal Component Analysis (PCA) to capture and plot meaningful variation.

## Limitations of Random Directions
- Ineffectiveness of Random Directions: Randomly chosen vectors fail to capture the variation in optimization trajectories, as observed by several authors.
- Failed Visualizations: Examples in Figure 8 demonstrate the inadequacy of capturing motion using random directions.

### Notable Attempts:
1. Projection onto Random Plane: Almost no motion captured, leading to seemingly random walk appearance.
2. One Direction from Initialization to Solution + One Random Direction: Random axis captures minimal variation, resulting in a misleading straight-line appearance.

## Exploring Low Dimensionality of Optimization Trajectories
- Randomly chosen vectors often lie orthogonal to low-rank spaces containing optimization paths.
- Utilizing PCA directions to validate low dimensionality and produce effective visualizations.
- Applying PCA to model parameter matrices at different epochs and selecting two most explanatory directions.

## Effective Trajectory Plotting with PCA Directions
- PCA provides a measure of how much variation is captured by each direction.
- Trajectories along PCA directions plotted on loss surfaces.
- Red dots indicate epochs where the learning rate was decreased.

The study introduces an approach based on PCA directions, providing an effective means to visualize optimization trajectories and measure the variation captured along each axis. This method overcomes the limitations of random directions, offering valuable insights into the dynamics of optimization paths.

# Dynamics of Optimization Paths: From Early to Later Stages

In the early stages of training, optimization paths tend to align perpendicular to the contours of the loss surface, following the expected behavior of non-stochastic gradient descent. However, as training progresses, the influence of stochasticity becomes more pronounced, particularly in scenarios involving weight decay and small batches.

## Training Dynamics Overview
- Early Training Behavior: Paths align perpendicular to loss surface contours, mirroring non-stochastic gradient descent.
- Later Training Stages: Increased stochasticity observed, especially in plots with weight decay and small batches.
- Effects of Weight Decay and Small Batches:
  - Parallel Movement: Paths turn nearly parallel to contours and exhibit an "orbit" around the solution, amplified by gradient noise in weight decay and small batches.
  - Stepsize Impact: Reducing stepsize (at red dot) decreases effective noise, causing a kink in the path as the trajectory converges into the nearest local minimizer.

## Dimensionality of Descent Paths
- Descent paths show very low dimensionality, with 40% to 90% of the variation confined to a space of only 2 dimensions.
- Optimization trajectories appear dominated by movement in the direction of a nearby attractor.
- Consistent with previous observations, where non-chaotic landscapes were characterized by wide, nearly convex minimizers.

The analysis provides insights into the evolving dynamics of optimization paths, highlighting the shift from early perpendicular alignment to later stages characterized by increased stochasticity and unique behaviors in the presence of weight decay and small batches. The low dimensionality observed aligns with previous landscape observations, reinforcing the connection between optimization path behavior and loss surface characteristics.