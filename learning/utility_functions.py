#### Define sparse neural network layer
"""
extended torch.nn module which cusmize connection.
This code base on https://pytorch.org/docs/stable/notes/extending.html
"""
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu,combine_pvalues
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
import seaborn as sns

#################################
# Define custome autograd function for masked connection.
class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True,dtype=torch.double,device="cpu"):
        """
        extended torch.nn module which mask connection.
        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(dtype).t()
        else:
            self.mask = torch.tensor(mask, dtype=dtype).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features).type(dtype).to(device))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features).type(dtype).to(device))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        nn.init.kaiming_uniform_(self.weight, mode="fan_in")
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


### Function to get a pytorch version of PLSR
class PLSRTransform(nn.Module):
    def __init__(self, X_coefs, X_mean, dtype=torch.double,device="cpu"):
        super(PLSRTransform, self).__init__()
        # Convert numpy arrays to torch tensors
        self.X_coefs = torch.tensor(X_coefs, dtype=dtype).to(device)
        self.X_mean = torch.tensor(X_mean, dtype=dtype).to(device)

    def forward(self, X):
        # Center the input X
        X_centered = X - self.X_mean
        
        # Project X onto the latent space (scores)
        Y_pred = torch.matmul(X_centered, self.X_coefs)
        
        return Y_pred
    
def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym,dim=0)
    x_square_sum = torch.sum(xm * xm,dim=0)
    y_square_sum = torch.sum(ym * ym,dim=0)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return r #torch.mean(r)

def numpy_pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    # Compute means for each column
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    # Subtract means (centering)
    xm = x - mx
    ym = y - my
    # Numerator: sum of element-wise products of centered vectors
    r_num = np.sum(xm * ym, axis=0)
    # Denominator: square root of the product of sum of squares of centered vectors
    r_den = np.sqrt(np.sum(xm * xm, axis=0) * np.sum(ym * ym, axis=0))
    # Pearson correlation for each column
    r = r_num / r_den
    return r

def stat_distribution_overlap(x,y,folds):
    x = np.asarray(x)
    y = np.asarray(y)
    # Initialize an array to store counts
    counts = np.zeros(x.shape)
    # Count elements in y that are greater than or equal to each element in x
    for i, val in enumerate(x):
        counts[i] = np.sum(y[i,:] >= val)
    p_value = np.sum(counts)/(len(x)*len(y))
    return p_value

def compare_target_vs_reference(df, genes2show,target_name, reference_name, gene_col='genes', model_col='set', r_col='r', folds_col='fold',alpha = 0.4):
    """
    Compares Pearson correlations across folds for shuffled and actual models, adjusts p-values,
    and plots a scatterplot.
    Parameters:
        df (pd.DataFrame): DataFrame containing columns for gene, set (e.g. shuffled/train/test), r values, and folds.
        genes2show (np.ndarray): 1D array of gene names to highlight on the scatterplot.
        gene_col (str): Column name for the gene.
        model_col (str): Column name for the model set.
        r_col (str): Column name for Pearson correlation values.
        folds_col (str): Column name for folds.
        target_name (str): Name of the target set (e.g. shuffled).
        reference_name (str): Name of the reference set (e.g. test).
    Returns:
        results_df (pd.DataFrame): DataFrame with gene, average r, p-value, and adjusted p-value.
        fig (plt.Figure): Scatterplot figure with avg r vs -log10(p-adjusted).
    """
    
    # Split the dataframe into shuffled and actual data
    df_shuffled = df[df[model_col] == target_name]
    df_actual = df[df[model_col] == reference_name]

    genes = df[gene_col].unique()
    results = []

    # Iterate through genes and perform t-tests across folds
    for gene in genes:
        r_shuffled = df_shuffled[df_shuffled[gene_col] == gene][r_col]
        r_actual = df_actual[df_actual[gene_col] == gene][r_col]
        
        # Perform t-test comparing shuffled and actual
        _, p_value = mannwhitneyu(r_actual, r_shuffled)
        
        # Compute average r across folds for actual model
        avg_r_actual = np.nanmean(r_actual)

        results.append({'gene': gene, 'avg_r': avg_r_actual, 'p_value': p_value})

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Adjust p-values using Benjamini-Hochberg correction
    results_df['p_adjusted'] = multipletests(results_df['p_value'], method='fdr_bh')[1]

    # Get the first two colors from the default Matplotlib color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_default = default_colors[0]  # First default color
    color_highlight = default_colors[1]  # Second default color
    # Set colors based on whether the gene will be highlighted cause it is important (in genes2show)
    colors = [color_highlight if gene in genes2show else color_default for gene in results_df['gene']]
    
    # Plot avg_r vs -log10(p_adjusted)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df['avg_r'], -np.log10(results_df['p_adjusted']), color=colors, alpha=alpha,s=15)
    ax.set_xlabel('Average r across folds')
    ax.set_ylabel('-log10(p-adjusted)')
    ax.set_title('Per feature performance')
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1.5)
    plt.tight_layout()

    return results_df, fig

def plot_combined_p_values(df, merged_results_df,target_name='shuffled',reference_name='test', set_col='set', r_col='r'):
    """
    Generates a boxplot comparing actual and shuffled models using combined p-values for the comparison.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame with Pearson correlation values for actual and shuffled models.
        merged_results_df (pd.DataFrame): DataFrame with adjusted p-values for each gene comparison.
        set_col (str): Column name for the model set (e.g., 'test', 'shuffled').
        r_col (str): Column name for Pearson correlation values.
        
    Returns:
        combined_p_value (float): Combined p-value across genes.
        fig (plt.Figure): Boxplot figure for the selected comparison.
    """
    
    # Step 1: Combine p-values using Fisher's method
    _, combined_p_value = combine_pvalues(merged_results_df['p_adjusted'], method='fisher')
    
    # Step 2: Filter DataFrame to include only test and shuffled sets
    df_filtered = df[df[set_col].isin([target_name, reference_name])]
    
    # Step 3: Visualize comparison of test vs shuffled with boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x=set_col, y=r_col, inner='box', ax=ax)
    ax.set_title('Comparison of Test vs Shuffled Models')
    ax.set_ylabel('Pearson’s Correlation')
    ax.set_xlabel('')

    # Annotate the plot with the combined p-value
    y_max = df_filtered[r_col].max() + 0.03  # Adjust y-axis for annotation
    x1 = np.where(df[set_col].unique() == reference_name)[0][0]
    x2 = np.where(df[set_col].unique() == target_name)[0][0]
    ax.text((x1 + x2) / 2, y_max, f'Combined p-value (Fisher) = {combined_p_value:.4e}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    return combined_p_value, fig

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, shared_layers, task_specific_layers, num_tasks,momentum=0.1, drop_rate=0.2, activation=nn.LeakyReLU(0.1), bias=True,dtype=torch.double):
        super(MultiTaskMLP, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        for layer_dim in shared_layers:
            self.shared_layers.append(nn.Linear(prev_dim, layer_dim, bias=bias, dtype=dtype))
            self.shared_layers.append(nn.BatchNorm1d(layer_dim, momentum=momentum, dtype=dtype))
            self.shared_layers.append(activation)
            self.shared_layers.append(nn.Dropout(drop_rate))
            prev_dim = layer_dim
        
        # Task-specific layers
        self.task_heads = nn.ModuleList()
        for _ in range(num_tasks):
            task_layers = nn.ModuleList()
            task_input_dim = shared_layers[-1] if shared_layers else input_dim
            for layer_dim in task_specific_layers:
                task_layers.append(nn.Linear(task_input_dim, layer_dim, bias=bias, dtype=dtype))
                task_layers.append(nn.BatchNorm1d(layer_dim, momentum=momentum, dtype=dtype))
                task_layers.append(activation)
                task_layers.append(nn.Dropout(drop_rate))
                task_input_dim = layer_dim
            task_layers.append(nn.Linear(task_input_dim, 1, bias=bias, dtype=dtype))  # Output layer for each task
            self.task_heads.append(task_layers)
    
    def forward(self, x):
        # Pass through shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Task-specific branches
        outputs = []
        for task_layers in self.task_heads:
            task_out = x
            for layer in task_layers:
                task_out = layer(task_out)
            outputs.append(task_out)
        
        return torch.cat(outputs, dim=1)
    
    def L2Regularization(self, L2_weight):
        l2_loss = 0.0
        # L2 regularization for shared layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                l2_loss += L2_weight * torch.sum(layer.weight ** 2)
                if layer.bias is not None:
                    l2_loss += L2_weight * torch.sum(layer.bias ** 2)
        
        # L2 regularization for task-specific layers
        for task_layers in self.task_heads:
            for layer in task_layers:
                if isinstance(layer, nn.Linear):
                    l2_loss += L2_weight * torch.sum(layer.weight ** 2)
                    if layer.bias is not None:
                        l2_loss += L2_weight * torch.sum(layer.bias ** 2)
        return l2_loss
    
class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, momentum=0.1, drop_rate=0.2, activation=nn.LeakyReLU(0.1), bias=True, dtype=torch.double):
        super(SimpleANN, self).__init__()
        
        # Feed-forward layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Hidden layers
        for layer_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, layer_dim, bias=bias, dtype=dtype))
            self.layers.append(nn.BatchNorm1d(layer_dim, momentum=momentum, dtype=dtype))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(drop_rate))
            prev_dim = layer_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim, bias=bias, dtype=dtype)
    
    def forward(self, x):
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        output = self.output_layer(x)
        
        return output
    
    def L2Regularization(self, L2_weight):
        l2_loss = 0.0
        
        # L2 regularization for all layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                l2_loss += L2_weight * torch.sum(layer.weight ** 2)
                if layer.bias is not None:
                    l2_loss += L2_weight * torch.sum(layer.bias ** 2)
        
        # L2 regularization for output layer
        if isinstance(self.output_layer, nn.Linear):
            l2_loss += L2_weight * torch.sum(self.output_layer.weight ** 2)
            if self.output_layer.bias is not None:
                l2_loss += L2_weight * torch.sum(self.output_layer.bias ** 2)
        
        return l2_loss

def train_model(X, Y,model, learning_rate, epochs, lamda_reg, batch_size,printing=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

    N = X.shape[0]
    mu_losses = []
    mu_r = []
    std_losses = []
    std_r = []
    model.train()
    for epoch in range(epochs):
        trainloader = getSamples(N, batch_size)
        batch_losses = []
        batch_r = []
        batch_sign_violations = []
        
        for i in range(len(trainloader)):
            dataIndex = trainloader[i]
            xin = X[dataIndex, :]
            yout = Y[dataIndex, :]
            optimizer.zero_grad()
            yhat = model(xin)
            fitLoss = torch.mean(torch.sum(torch.square(yhat - yout), dim=1)) 
            regLoss = model.L2Regularization(L2_weight=lamda_reg)
            # masked_weight = reg_model.weight.T * torch.abs(mask_signed)
            # signViolation = masked_weight * mask_signed
            # singLoss = sign_reg * torch.sum(signViolation < 0)
            loss = fitLoss + regLoss
            
            loss.backward()
            optimizer.step()
            
            r = pearson_r(yout, yhat).mean()
            batch_losses.append(loss.item())
            batch_r.append(r.item())
        scheduler.step()
        mu_losses.append(np.nanmean(batch_losses))
        mu_r.append(np.nanmean(batch_r))
        std_losses.append(np.nanstd(batch_losses))
        std_r.append(np.nanstd(batch_r))
        
        if ((epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == epochs) and printing:
            print('Epoch: {}/{}, r: {:.4f}, Loss: {:.4f}'.format(epoch + 1, epochs, np.nanmean(batch_r), np.nanmean(batch_losses)))
    
    model.eval()
    with torch.no_grad():
        yhat = model(X)
        r = pearson_r(Y, yhat)
    
    return model,mu_losses, mu_r, std_losses, std_r, r.detach().cpu().numpy()