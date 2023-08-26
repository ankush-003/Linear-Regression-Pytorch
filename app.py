import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr



def get_error(data, a, b):
    n = data.size()[0]
    y_pred = a * data[:,0] + b
    #return (1 / (2 * n)) * ((y_pred - data[:,-1]) ** 2).sum().item()
    return (1 / 2) * torch.mean((y_pred - data[:,-1]) ** 2).item()

def get_grads(data, a, b, alpha):
    n = data.size()[0]
    y_pred = a * data[:,0] + b
    #grad_a = (alpha / n) * ((y_pred - data[:,-1])* data[:,0]).sum().item()
    grad_a = alpha * torch.mean((y_pred - data[:,-1])* data[:,0]).item()
    #grad_b = (alpha / n) * (y_pred - data[:,-1]).sum().item()
    grad_b = alpha * torch.mean((y_pred - data[:,-1])* data[:,0]).item()
    return (grad_a, grad_b)

def train(data, alpha=0.0001, epochs=500, test_data=[1,2,3]):
    tensors = torch.from_numpy(data.astype(np.float64))
    print(f"dataset:{tensors}, size:{tensors.size()}")
    a = 1
    b = 1
    errors = []
    updated_a = []
    updated_b = []
    for i in tqdm_notebook(range(int(epochs)), desc=f"Epoch: "):
        errors.append(get_error(tensors, a, b))
        grad_a_temp, grad_b_temp = get_grads(tensors, a, b, alpha)
        updated_a.append(a)
        updated_b.append(b)
        a -= grad_a_temp
        b -= grad_b_temp
        
        #print(f"Epoch: {i} -> Theta0: {updated_b[i]} Theta1: {updated_a[i]}, Error: {errors[i]} \n")
    y_pred = a * tensors[:,0] + b 
    fig1 = px.scatter(x=tensors[:, 0], y=tensors[:, -1], template="plotly_dark")
    fig2 = px.line(x=tensors[:, 0], y=y_pred, template="plotly_dark") 
    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_layout(
        #title="Best Fit Line",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_dark"
    )

    fig4 = px.line(errors, template="plotly_dark")
    fig4.update_traces(line_color='orange')    
    fig4.update_layout(
        #title="Learning Curve",
        xaxis_title="Epoch",
        yaxis_title="Error",
        showlegend=False
    )
    y_pred_test = a * test_data.astype(np.float64) + b 
    return f"{round(updated_a[-1],3)} * X + {round(updated_b[-1],3)}",fig3, fig4, np.concatenate((test_data.astype(np.float64), y_pred_test), axis=1)

# gradio app

input_elements = [
    gr.Numpy(
            value=[[40,9],[30,8.5],[25,8],[20,7],[10,6],[5,8]],
            datatype="number",
            row_count=(6,"dynamic"),
            col_count=(2, "fixed"),
            label="Dataset",
            interactive=True
        ),
    gr.Number(label="Learning Rate", value=0.0001, interactive=True),
    gr.Number(label="Number of epochs", value=500, interactive=True),
    gr.Numpy(
        value=[[10],[20],[30]],
        datatype="number",
        row_count=3,
        col_count=(1,"fixed"),
        label="Test Dataset",
        interactive=True
    )
]

output_elements = [
    gr.Textbox(label="Generated Model", placeholder="Model Equation: a * X + b"), 
    gr.Plot(label="Best Fit Line"), 
    gr.Plot(label="Learning Curve"),
    gr.Numpy(label="Model Predictions")
    
]

app = gr.Interface(
    title="Linear Regression using Pytorch",
    #description="a simple app to demonstrate linear regression",
    fn=train,
    inputs=input_elements,
    outputs=output_elements,
    allow_flagging="never"
)
app.launch(debug=True)