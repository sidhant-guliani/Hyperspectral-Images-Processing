# Hyperspectral-Images-Processing
    Plotting 10 of the random bands in the data. there are totla of 200 band for 145*145 pixel image.
![](/plots/image_rawdata.png)

    Ground visualization for 16 different classification in the data. For the model preparation we are going to remove 
    class 0: Unknown(in black)
![](/plots/gt_viualization.png)

    Adding three of the random bands together to get the image: RGB composite image.
![](/plots/rgb_composite_image.png)


    Adding three of the random bands together to get the image: RGB composite image.
![](/plots/spectral_plot.png)

    Plot to see the general shape of spectra for each class, we are doign this to check the general shape of spectra for each class. 
    The bold line in each plot is the average of all the spectral lines for that class.
![](/plots/class_wise_spectra.png)

    The plot for the validation/training accuracy and loss. Model explained later.
![](/plots/CNN_model.png)

    Confusion matrix, The accuracy is roughly 86%, we can improve out model by doing PCA or working on 2D/3D CNN. Please see below
![](/plots/CNN_model2.png)

