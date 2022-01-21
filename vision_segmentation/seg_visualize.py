import  rasterio
from rasterio.enums import Resampling
import numpy  as  np
import torch
from typing import  Optional, Sequence,Dict,Union
from torch import Tensor
from matplotlib import colors
import matplotlib.pyplot as plt

def percentile_normalization(
    img: "np.typing.NDArray[np.int_]",
    lower: float = 2,
    upper: float = 98,
    axis: Optional[Union[int, Sequence[int]]] = None,
) -> "np.typing.NDArray[np.int_]":
    """Applies percentile normalization to an input image.

    Specifically, this will rescale the values in the input such that values <= the
    lower percentile value will be 0 and values >= the upper percentile value will be 1.
    Using the 2nd and 98th percentile usually results in good visualizations.

    Args:
        img: image to normalize
        lower: lower percentile in range [0,100]
        upper: upper percentile in range [0,100]
        axis: Axis or axes along which the percentiles are computed. The default
            is to compute the percentile(s) along a flattened version of the array.

    Returns
        normalized version of ``img``

    .. versionadded:: 0.2
    """
    assert lower < upper
    lower_percentile = np.percentile(img, lower, axis=axis)
    upper_percentile = np.percentile(img, upper, axis=axis)
    img_normalized: "np.typing.NDArray[np.int_]" = np.clip(
        (img - lower_percentile) / (upper_percentile - lower_percentile), 0, 1
    )
    return img_normalized


def _load_image( path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image
            shape: the (h, w) to resample the image to

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
            )
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor
def _load_target(path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_dtype="int32", resampling=Resampling.bilinear
            )
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
            return tensor


def  get_sample(path:str) ->Dict:
    #原图   多通道
    image = _load_image(path+"/44-2013-0286-6713-LA93-0M50-E080.tif")
   
    #高度图   float32存储
    dem =   _load_image(path+"/44-2013-0286-6713-LA93-0M50-E080_RGEALTI.tif",shape = image.shape[1:])
    #dem = None
    #target  单通道
    target= _load_target(path+"/44-2013-0286-6713-LA93-0M50-E080_UA2012.tif")

    return {"image":image,"dem":dem,"mask":target}

def plot(sample: Dict[str, Tensor],show_titles: bool = True,suptitle: Optional[str] = None) -> plt.Figure:
    """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
    """
    colormap = [
        "#231F20",
        "#DB5F57",
        "#DB9757",
        "#DBD057",
        "#ADDB57",
        "#75DB57",
        "#7BC47B",
        "#58B158",
        "#D4F6D4",
        "#B0E2B0",
        "#008000",
        "#58B0A7",
        "#995D13",
        "#579BDB",
        "#0062FF",
        "#231F20",
    ]
    ncols = 2
    image = sample["image"][:3]
    image = image.to(torch.uint8)  # type: ignore[attr-defined]
    image = image.permute(1, 2, 0).numpy()

    dem = sample["dem"].numpy()
    dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1)).squeeze()
    print(dem.shape)

    showing_mask = "mask" in sample
    showing_prediction = "prediction" in sample

    cmap = colors.ListedColormap(colormap)

    if showing_mask:
        mask = sample["mask"].numpy()
        ncols += 1
    if showing_prediction:
        pred = sample["prediction"].numpy()
        ncols += 1

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

    axs[0].imshow(image)
    #关闭子图中的轴
    axs[0].axis("off")
    axs[1].imshow(dem)
    axs[1].axis("off")
    if showing_mask:
        axs[2].imshow(mask, cmap=cmap, interpolation=None)
        axs[2].axis("off")
        if showing_prediction:
            axs[3].imshow(pred, cmap=cmap, interpolation=None)
            axs[3].axis("off")
    elif showing_prediction:
        axs[2].imshow(pred, cmap=cmap, interpolation=None)
        axs[2].axis("off")

    if show_titles:
        axs[0].set_title("Image")
        axs[1].set_title("DEM")

        if showing_mask:
            axs[2].set_title("Ground Truth")
            if showing_prediction:
                axs[3].set_title("Predictions")
        elif showing_prediction:
            axs[2].set_title("Predictions")

    if suptitle is not None:
        plt.suptitle(suptitle)

    return fig

if __name__ == "__main__":

    sample = get_sample("./test")
    fig = plot(sample,suptitle = "Semantic segmentation of remote sensing")
    plt.show()

        
