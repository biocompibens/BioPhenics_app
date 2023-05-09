"""should rassemble all function for manipulating images"""
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
import re

from PIL import Image
import numpy as np
from pandas import DataFrame

from scripts.global_name import PATH, BARCODE, WELLS, FIELDS, WAVE


class ScreeningImage(object):
    """
    class utility for manipulating images
    """
    is_loaded = False
    _array = None  # lazy loading
    _image = None
    _mode = 'L'

    def __init__(self, path=None, barcode=None, wells=None,
                 field=None, wavelength=None, mode="L", prefix_name=""):
        """
        construct an instance, with path as the path of image
        (it is lazy loaded : until we access data, image is not opened)
        and metadata
        Parameters
        ----------
        path: pathlike
            image path
        barcode: str
            identification number of the images plate
        wells: str
            images wells
        field: str or int
            images field
        wavelength: str or int
            images wavelength (or channels) better support for wavelength as it
            allow to merge them
        mode: str
            images mode (value can be "I;16", "L" or "RGB"), it describe image
            type
                * "I;16" => image encoded 16 bits default output for incell 6500
                  and consort (.tif)
                * "L" => image encoded in 8 bits, lower quality but much
                  less spaces is taken and can be manipulated more easily by
                  Pillow
                * "RGB" => same encodage as "L" but with 3 channels red,
                  green and blue
        """
        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError

        self.path = path
        self.barcode = barcode
        self.wells = wells
        self.field = field
        self.wavelength = wavelength
        self._mode = mode
        self.filename = prefix_name

    @classmethod
    def from_series(cls, serie, mode='L'):
        """"""
        result = cls(mode=mode)
        for attr_name, col_name in [("path", PATH), ("barcode", BARCODE),
                                    ("wells", WELLS), ("field", FIELDS),
                                    ("wavelength", WAVE)]:
            if col_name in serie:
                try:
                    setattr(result, attr_name, serie[col_name].squeeze())
                except AttributeError:
                    setattr(result, attr_name, serie[col_name])
        return result

    def save(self, filelike=None, **kwargs):
        """
        Save image to file-like and return it

        Parameters
        ----------
        filelike: file object or path to where to record it
        kwargs: options to pass along to the pillow save method

        Returns
        -------
        the path or file-like object that recorded image

        """
        if self.is_loaded:
            filelike = filelike or self.path
            try:
                self.image.save(filelike, **kwargs)
            except OSError as e:
                if self._mode in str(e):
                    self.convert("L")
                    self.image.save(filelike, **kwargs)
                else:
                    raise
            if isinstance(filelike, BytesIO):
                filelike.seek(0)
            return filelike
        return self.path

    def close(self):
        try:
            return self.image.close()
        finally:
            self.is_loaded = False

    @classmethod
    def merge(cls, r, g, b, mode="RGB"):
        """
        given three mono-channels images, merge them into one newly created
        RGB images

        Parameters
        ----------
        r: image
             red channel
        g: image
            green channel
        b: image
            blue channel
        mode:
            mode of the resulting image (default 'RGB')

        Returns
        -------
        the resulting image

        """
        merged = cls()
        merged.image = Image.merge(mode, [r, g, b])
        return merged

    def convert(self, *args, **kwargs):
        self.image = self.image.convert(*args, **kwargs)

    def _lazy_load(self):
        """
        open image when we want to access it

        Returns
        -------
        None
        """
        if self._image is None:
            if self._array is not None:
                self._image = Image.fromarray(self._array)
            else:
                try:
                    self._image = Image.open(self.path)
                except (OSError, AttributeError):
                    raise OSError(f'Image "{self.path}" not found')
            self.is_loaded = True
            self._mode = self._image.mode

    @property
    def image(self):
        self._lazy_load()
        return self._image

    @image.setter
    def image(self, img):
        if hasattr(img, "mode"):
            self._mode = img.mode
        self._image = img
        self.is_loaded = True

    @property
    def array(self):
        if self._array is None:
            self._array = np.asarray(self.image)
        return self._array

    @array.setter
    def array(self, array, mode=None):
        self._array = array
        try:
            self._image = Image.fromarray(
                self._array
            ).convert(mode or self._mode)
        except TypeError:
            self._image = Image.fromarray(self._array, "RGB")
        self.is_loaded = True
        if len(self._array.shape) != 2:
            self._mode = "RGB"

    @property
    def name(self):
        if not self.filename and self.path is not None:
            return self.path.stem
        filename = self.filename
        parenthesis = False
        if self.barcode:
            filename += f" {self.barcode}"
        if self.wells:
            if filename:
                filename += ' - '
            filename += f"{self.wells}"
        if self.field:
            if filename:
                filename += " ("
                parenthesis = True
            filename += f"fld {self.field}"
        if self.wavelength:
            if filename:
                filename += " "
            filename += f"wv {self.wavelength}"

        if parenthesis:
            filename += ')'

        return filename

    # @property
    # def ext(self):
    #     if not self.is_loaded:
    #         return self.path.suffix
    #     else:
    #         if self.path is not None and self.path.suffix != 'tif':
    #             return self.path.suffix
    #         return 'jpeg'  # default

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        if value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                try:
                    value = re.search(r'\d+', value).group(0)
                except AttributeError:
                    raise ValueError('Wrong type of wavelength')
        self._wavelength = value

    def normalize(self, low=None, high=None):
        """
        Usually I;16 image distribution do not take all range
        (value from 0 to 65536) but a small part. When we want to convert it
        (or anything else actually) we first need to normalize it by
        transforming this distribution into a 0 to 255 range.
        We take the first and last percentile as min and max of the
        distribution and we normalize by the formula : ((X - low) / high) * 255
        We cap all values outside [0;255] at this interval extremum

        Parameters
        ----------
        low: value for distribution minimal
        high: value for distribution maximal

        Returns
        -------
        None, the array is modified in place
        """
        low = low or np.percentile(self.array, 1)
        high = high or np.percentile(self.array, 99)
        self.array = ((self.array - low) * (255 / high))
        self.array[self.array > 255] = 255
        self.array[self.array < 0] = 0

    # def thumbnail(self, size=()):
    #     """
    #     make use of pillow thumbnails method
    #
    #     Parameters
    #     ----------
    #     size: tuple of int
    #         size of the resulting thumbnails
    #
    #     Returns
    #     -------
    #     None, image is modified in place
    #
    #     """
    #     res = self.image
    #     res.thumbnail(size=size)
    #     self.image = res

    @property
    def size(self):
        return self.image.size

    # def to_series(self):
    #     return pd.Series({
    #         col_name: getattr(self, attr_name)
    #         for attr_name, col_name in [
    #             ("path", PATH), ("barcode", BARCODE),
    #             ("wells", WELLS), ("field", FIELDS),
    #             ("wavelength", WAVE)
    #         ]
    #     })


class ListImage(object):
    """
    Classes that handle a list of ScreeningImage
    """

    def __init__(self, iterable=None):
        """
        Construct an instance from a list (or any iterable) of ScreeningImage

        Parameters
        ----------
        iterable: iterable of ScreeningImage

        """
        try:
            iterator = iter(iterable)
        except TypeError:
            self.images = DataFrame(
                columns=['barcode', 'wells', 'field', 'wavelength', 'img']
            )
        else:
            if not all([isinstance(i, ScreeningImage) for i in iterator]):
                raise ValueError('Wrong type, need a iterable of '
                                 'ScreeningImage Object')
            self.images = DataFrame(
                [[i.barcode, i.wells, i.field, i.wavelength, i]
                 for i in iterable],
                columns=['barcode', 'wells', 'field', 'wavelength', 'img'])

    # def __iter__(self):
    #     yield from self.images['img']

    # def __len__(self):
    #     return self.images.__len__()

    # def __getitem__(self, item):
    #     return self.img[item]

    @property
    def names(self):
        return [i.name for i in self.img]

    @property
    def barcode(self):
        return self.images['barcode']

    @property
    def wells(self):
        return self.images['wells']

    @property
    def field(self):
        return self.images['field']

    @property
    def wavelength(self):
        return self.images['wavelength']

    @property
    def img(self):
        return self.images['img']

    def groupby(self, *args, **kwargs):
        for grp_name, grp in self.images.groupby(*args, **kwargs):
            new_list = ListImage()
            new_list.images = grp
            yield grp_name, new_list

    def normalize(self, by=None):
        """
        normalize pixel value for all image, if by is set a common value for the
        group by is used to normalise grouped image

        Parameters
        ----------
        by: str
            name for groups

        Returns
        -------
        None, images are normalized inplace

        """
        if by is None:
            grps = zip_longest([], [ListImage([i]) for i in self.img])
        elif by == 'all':
            grps = [(None, self.images)]
        else:
            grps = self.images.groupby(by)

        for name, imgs in grps:
            low, high = np.percentile([i.array for i in imgs.img], [1, 99])
            for img in imgs.img:
                img.normalize(low, high)
