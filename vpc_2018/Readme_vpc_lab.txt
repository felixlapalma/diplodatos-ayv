Casos
*****

Se corrieron los siguientes casos (la descripcion no es extensiva sino orientativa):

Experiment001 - Full Set (Train_frac 0.8 - val_frac 0.2) - categorical_cross_entropy - dropout 0.5 - fit on layer >87
Experiment002 - Full Set (Train_frac 0.8 - val_frac 0.2) -  SGD - dropout 0.25 - fit on layer >87
Experiment003 - Reduced Set (Train_samp 20000 - val_frac 0.2) - categorical_cross_entropy - dropout 0.5 - fit on layer >87
Experiment004 - Reduced Set (Train_samp 20000 - val_frac 0.2) - categorical_cross_entropy - dropout 0.1 fit on layer >85
Experiment005 - Reduced Set (Train_samp 20000 - val_frac 0.2) - categorical_cross_entropy - dropout 0.75 fit on layer >85
Experiment006 - Full Set (Train_frac 0.6 - val_frac 0.4) - categorical_cross_entropy - dropout 0.75 - fit on layer >87
Experiment007 - Full Set (Train_frac 0.6 - val_frac 0.4) - SGD - dropout 0.25 - fit on layer >85

Cada caso tiene su ipynb asociada, junto a los registrado por los callbacks TensorBoard y ModelCheckpoint.

La eleccion se baso en los logs de Tensorboard. En funcion de estos se opto por el Experiment001.
-Valor de accuracy en (train,val)
-Valor de loss en (train,val)  (hasta la etapa 6 - si se observa los logs)

Algunos comentarios en orden, si bien no se sumaron mas capas que 3 (tres), se probaron
diferentes valores de configuracion para DropOut y Balance de Train y Validation. Asimismo como capas entrenables.
Da la impresion que algo estoy realizando en forma No correcta puesto que los valores de accuracy en validacion se mantienen estables.

Los archivos principales se suben a github, entre ellos:
-Lab_VpC_FelixRojoLapalma_TEST.ipynb (donde se evaluo el modelo final)
-results_MobileNet.csv (comparacion de resultados)
-experiment_XXX se dejaron los resultados de guardados por tensorboard para visualizacion (si se requiere)

El conjunto total se puede descargar desde: 
https://drive.google.com/open?id=1NZb-OjbkxgV-__uiRGVJjINePYVEyg-g


