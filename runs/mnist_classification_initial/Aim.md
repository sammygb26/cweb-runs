# `MNIST` Classification Initial
Currently the `cweb` library has been implemented up to the point of **reuploading** for fixed layers in the clement's scheme. Along with this some extra functionality (`AffineLayer`, `softmax` and `cross_entropy`) has been implemented to support a classifier implementation.

For this simple case we just want to verify it is training. Laster *hyperparameter tuning* can be perfornmed. Below is the scheme where a specific quantum states `|p>` is passed through a clements scheme generated from a reuploading scheme. The probability distribution of the resulting mixed state has an affine transformation applied before a softmax. This gives the output probability distribution. Softmax is then used to create a loss which is finally fed into Adam to perform parameter updates.

```
┌────┐    ┌────────┐    ┌─────┐                                                         
│DATA├───►│BATCHING├───►│BATCH├───────┐   |p>                                           
└────┘    └────────┘    └─────┘       │    │                                            
                                     ┌▼────▼──┐                                         
                                     │CLEMENTS│                                         
                     ┌────────┐      └▲────┬──┘                                         
                     │CLEMENTS│       │    │                                            
                     │REUPLOAD├───────┘ ┌──▼──┐    ┌─────────┐    ┌───────┐    ┌───────┐
                     │ SCHEME │         │MIXED├──► │ AFFINE  ├───►│SOFTMAX├───►│CROSS  │
                     └───▲────┘         │STATE│    │TRANSFORM│    └───────┘    │ENTROPY│
                         │              └─────┘    └────▲────┘                 └───┬───┘
                         │                              └─UPDATES──┬────┐          │    
                         │                                         │ADAM│◄──LOSS───┘    
                         └──────────────UPDATES────────────────────┴────┘               

```
