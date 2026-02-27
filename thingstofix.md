1. utilizzo non corretto di PointnetConv su PyG.
2. farmi spiegare la vettorizzazione dei calcoli
3. aumentare profondita' pointnet
4. splittare in validation set e training
5. aggiungere accuracy e mIoU
6. aggiungere scheduler

avevo creato "load and downsample pointcloud by skip" perche', con il voxel grid downsampling, la nuvola risultante era composta da punti che erano completamente diversi rispetto a quella originale.

Questo diventava un problema perche' faccio il downsampling dell'intera nuvola prima di fare i calcoli delle normali e curvatura, e la nuvola risultante e' composta da punti diversi che vanno a "mescolare" l'appartenenza di ciascun punto a una determinata superficie.

- questo problema forse si puo' fixare in maniera piu' "elegante" facendo il downsample della nuvola non per intero, ma superficie per superficie. in questo modo i punti che ne risulteranno saranno si' diversi, ma sicuramente "seguiranno lo stesso trend" facendo parte della stessa superficie.
  - l'unica cosa che mi preoccuperebbe e' che, potenzialmente, la pointcloud si "restringerebbe" dato che i centroidi sono appunto piu' al centro, e dato che e' applicata in maniera separata su tutte le superfici, ci sarebbe un "gap" nei punti di giuntura fra le superfici, che non sarebbe altro che un artefatto del mio algoritmo che non e' minimamente riflesso poi nella realta'. Nel momento in cui andiamo a fare il downsampling su una nuvola normale in momento di inferenza, non abbiamo idea di quali siano le superfici, e facendolo sull'intera nuvola al posto di superficie per superficie, questo artefatto non si presenterebbe, creando un'asimmetria in come trattiamo i dati.
  - da vedere se questa preoccupazione e' legittima.

In che modo il metodo "by skip" distrugge l'uniformita' dei punti? da quello che ricordo semplicemente ne prendeva 1, ne saltava x, poi ne prendeva un'altro etc..., non e' che avesse un criterio che potesse discriminare i bordi.

---

### Cose da chiedere a Buonacucina

- parlargli di questa preoccupazione del voxel downsample

---

## Cose fatte

- aggiunta pezzo per trainare su GPU
- aggiunta normalizzazione del dataset
- vettorizzazione dei calcoli per la discriminative loss
- confermato che gli id sono UNICI per ogni superficie, indipendentemente dalle ripetizioni.
- cleaned up some imports
