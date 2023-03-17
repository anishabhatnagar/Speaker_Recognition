# Speaker_Recognition

#### Aim
Recognizing a speaker by his/her/their voice by processing audio signals and extracting relevant features. 


#### Dataset
Speaker Recognition Dataset from kaggle was selected.
* This dataset contains speeches of these prominent leaders; Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret Thatcher and Nelson Mandela which also represents the folder names. 
* Each audio in the folder is a one-second 16,000 sample rate PCM encoded.
* The speeches were collected from the  American Rhetoric (online speech bank).
* Each label has 1,500 samples. 
* The size of the entire dataset is 7,500 samples.

#### Classic spectrograms 
* Classic spectrograms were generated using torchaudio.transforms.spectrogram() function from the pytorch audio library.
* On training of the Resnet-18 model , with the tensor dataset and the regularisation techniques (SGD optimizer), with a learning rate between 1e-3 and 1e-5, 200 epochs, dropout probability between 0.4 and 0.8 and weight decay set to 0.01, we got 92.6% testing accuracy.

#### Mel Spectrograms 
* Mel Spectrograms were generated from torchaudio.transforms.MelSpectrogram() from the pytorch audio library. 
* On training of the Resnet-18 model , with the tensor dataset and the regularisation techniques (SGD optimizer), with a learning rate between 1e-3 and 1.5e-5, 325 epochs, dropout probability between 0.4 and 0.8 and weight decay set to 0.01, we got 93.75% testing accuracy.

#### MFCCs 
* MFCC features are the most commonly used features for speaker recognition.
* MFCC features were generated using torchaudio.transforms.MFCC() function from the PyTorch audio library.
* Once again, on training of the Resnet-18 model , with the tensor dataset and the regularisation techniques (SGD optimizer), with a learning rate between 1e-3 and 1e-4, 100 epochs, dropout probability between 0.4 and 0.8 and weight decay set to 0.01, we got 95.3%  testing accuracy.


