import unicodedata
from re import search, I
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from pandas import merge, DataFrame
import numpy as np

from scripts.global_name import (
    WELLS, COL_ORDERED, PLATE, CONTENT, ROW,
    COLUMN, LIGNEE, CLUSTER,
    FEATURE_NAME_PREFIX_DEEP_LEARNING,
)


def get_commun_ctrl(df, ctrl):
    """
    find control wells that are present in all plate of a DataFrame df

    Parameters
    ----------
    df: DataFrame
        DataFrame of a project arranged as in prenormalization.data object
    ctrl: list of str
        names of the controls

    Returns
    -------
        DataFrame of controls wells common on all plate
    """
    ctrl_df = df.loc[df[CONTENT].isin(ctrl)]
    common = set()
    for _, plate_ctrl in ctrl_df.groupby(PLATE):
        if not common:
            common = set(plate_ctrl[WELLS])
        else:
            common = common.intersection(set(plate_ctrl[WELLS]))
    return ctrl_df.loc[ctrl_df[WELLS].isin(common)]


def join_and(items):
    """
    same as join with ', ' as separator and ending with 'and'

    Parameters
    ----------
    items: list of str
        list of piece of string to be joined
    Returns
    -------
    the resulting string
    """
    return ' and '.join(', '.join(items).rsplit(", ", 1))


def clear_compound_name(text, lower=True):
    """
    perform a number of task to make compound name consistent accross different
    project
    notably :
    * strip number of non word characters
    * lowercase
    * transform all accent into their corresponding accentless characters

    Parameters
    ----------
    text: str
        compound name

    lower: bool
        if it's True, all characters will be converted to lowercase

    Returns
    -------
        compound name transformed
    """
    if text is np.nan:
        return ''
    if lower:
        text = str(text).lower()
    if text == 'nan' or text == 'none' or text == 'null':
        return ''

    text = text.strip('"\',;.:/!?<>~#{[]}=+°^\\_`-| \t\n\r§')
    # remove trailing useless caractere

    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode("utf-8")
    # transform accent letter to their non accented version

    return text


def form_well(pos):
    """Format the wells (remove trailing space,
    add uppercase and add 0 (ex: a1 => A01, B - 22 => B22...)"""
    try:
        res = search(r"([a-zA-Z])[\s\-_]*(\d{1,2})(?:.*\(fld\s*(\d))?",
                     pos, flags=I)
        if res:
            puit = res.group(1).upper() + f"{int(res.group(2)):02d}"
            if res.group(3) is not None:
                return puit + ";" + res.group(3)
                # si pos correspond a un champs,
                # renvoie aussi le numero du champs
            return puit
        return pos
    except (AttributeError, IndexError, TypeError):
        return pos


def merge_rep(dataframe, group_by, on_cols, how='outer', nafill=None):
    """
    Take each group from group_by and merge each on_cols on one row

    Parameters
    ----------
    dataframe: DataFrame
        the data in pandas' dataframe

    group_by: String
        Column name of the dataframe, it will be grouped by this columns value

    on_cols: List of string
        Columns that do not change their value in each group_by group

    how: basestring
        the way the merge is performed (value can be 'inner' 'outer')
        (see pandas merge function for more detailed information)

    nafill: Object
        fill na with naFill after the merge if some column and or row are NaN

    Returns
    -------

    outdf: DataFrame
        Merged dataframe

    """
    outdf = None
    for repl, donnee in dataframe.groupby(group_by, sort=False):
        df0 = donnee.copy()
        cols = donnee.columns
        specific_cols = [x for x in cols if x not in on_cols and x != group_by]
        df0 = df0[on_cols + specific_cols]
        df0.columns = on_cols + [(s + '_' + repl) for s in specific_cols]

        if outdf is None:
            outdf = df0
        else:
            outdf = merge(outdf, df0, how=how, on=on_cols, validate="1:1")

    if nafill is not None:
        outdf = outdf.fillna(nafill)

    return outdf


def sort_column(df, index=False):
    """Trie les colonnes selon l'ordre 'classique',
    renvoie le nom des colonne si index est vrai sinon
    un nouveau df reordonné"""
    column = set(df.columns)
    first_part = [col for col in COL_ORDERED if col in column]
    order = first_part + sorted(list(column - set(first_part)))
    return order if index else df.reindex(order, axis=1)


def extract_row_and_col(serie):
    """
    From a serie of wells return a dataframe with the row and columns extracted

    Parameters
    ----------
    serie: Pandas Series
        A pandas series with Wells in the good format (:func:`form_wells`)

    Returns
    -------
    Pandas DataFrame
        A dataframe with two columns one with the name of row
        and one with the name of columns the index of the input
        is preserved in the output

    Notes
    -----

    +-----+-------++----------------------+
    | input:      || output:              |
    +-----+-------++-----+------+---------+
    |     | Wells ||     | Row  | Columns |
    +-----+-------++-----+------+---------+
    | 0   | A01   || 0   | A    |  1      |
    +-----+-------++-----+------+---------+
    | 1   | A02   || 1   | A    |  2      |
    +-----+-------++-----+------+---------+
    | ... | ...   || ... | ...  |  ...    |
    +-----+-------++-----+------+---------+
    | 382 | P23   || 382 | P    |  23     |
    +-----+-------++-----+------+---------+
    | 383 | P24   || 383 | P    |  24     |
    +-----+-------++-----+------+---------+

    """
    resultat = serie.str.extract(fr"(?P<{ROW}>\w)(?P<{COLUMN}>\d\d)")
    resultat[COLUMN] = resultat[COLUMN].astype(int)
    return resultat


def to_array(df, column, numpy=True, coerce_float=True, formats=384):
    """
    Construit une array 16 * 24 (384 wells plate)
    a partir d'un dataframe contenant 2 colonnes :
    * coordonnées du puit
    * column
    """
    if len(df) > formats:
        raise ValueError('Plate have invalid number of wells')

    data2 = df[[column]]
    row_col = extract_row_and_col(df[WELLS])
    letters = list(str_range('Q' if formats == 384 else 'I'))
    numbers = list(range(1, 25 if formats == 384 else 13))
    data2.index = [row_col[ROW].values, row_col[COLUMN].values]
    result = DataFrame(data2, index=[
        letters * len(numbers),
        [j for n in numbers for j in [n] * len(letters)]
    ])
    result = result.unstack().droplevel(0, axis=1)

    if numpy:
        result = result.values

    if coerce_float:
        return result.astype(float)
    return result


def from_array(array, name='feature'):
    """
    take an numpy array which represent a 384-wells or
    96 plate and return a df with 2 col : WELLS and the feature
    :param array: a numpy array
    :param name: the column name
    :return: a pandas dataframe
    """
    df = DataFrame(array)
    df = df.unstack()
    wells = DataFrame(df.index, columns=[WELLS])
    wells = wells.applymap(
        lambda x: f"{chr(65 + int(x[1]))}{int(x[0]) + 1:02d}"
    )
    df.index = wells.index
    res = wells.assign(**{name: df})
    res = res.sort_values(by=WELLS)
    res.index = range(len(res))
    return res


def plate_and_back(my_func):
    """Decorator for a function which can be use on a plate-like df"""

    def modif(df, feature):
        old_index = df.index
        new_df = from_array(my_func(to_array(df, feature)), feature)
        if len(new_df) != len(old_index):  # some wells are missing
            new_df = new_df.dropna()  # can't have missing wells and Na values
        new_df.index = old_index
        return new_df

    return modif


def str_range(l1, l2=None, step=1):
    """
    As built-in range, this function return a range of letter (from A to Z)
    """
    if l2 is None:
        l2, l1 = l1, 'A'
    l1, l2 = l1.upper(), l2.upper()
    return (chr(i) for i in range(ord(l1), ord(l2), step))


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_all_child_from_both_side(cluster_idx, matrix, nb_leaf, selected):
    """
    Recursive function to get all leaves from the distance matrix
    It allow us to get the cluster id for each one
    (and not the intermediary nodes)

    Parameters
    ----------
    cluster_idx: int
        current cluster identification number
    matrix: numpy array
        distance matrix
    nb_leaf: int
        total number of leaves
    selected: dict
        allow us to make sure that we don't pass twice by the same node

    Returns
    -------
    all child from cluster_idx

    """
    cluster_idx -= nb_leaf
    if cluster_idx in selected:
        return []

    left = int(matrix[cluster_idx, 0])
    right = int(matrix[cluster_idx, 1])

    if left >= nb_leaf:
        left = get_all_child_from_both_side(left, matrix, nb_leaf, selected)
        try:
            left = left[0] + left[1]
        except IndexError:
            pass
    else:
        left = [left]

    if right >= nb_leaf:
        right = get_all_child_from_both_side(right, matrix, nb_leaf, selected)
        try:
            right = right[0] + right[1]
        except IndexError:
            pass
    else:
        right = [right]

    return left, right


def compute_cluster(hits, features, color_threshold=None):
    """
    Calculate the cluster the same way as in :func:`create_dendrogram`
    (didn't find any simpler way to get this result other than rewrite the
    function...)

    Parameters
    ----------

    hits: DataFrame
        the list of hits
    features: str or list of str
        column name of hits that will be used to calculate
        the distance between hits
    color_threshold: float
        float between 0 and 1, it represent normalized distance at which
        clusters are colored differently (0: all leaves are colored differently,
        1: only one color) default = 0.7

    Returns
    -------
    Series
    A Series with a class for each cluster (each classes is represented by a int
    between 1 and len(classes))

    """
    def by_lines(df, threshold=None):
        df = df.round(10).groupby(CONTENT).agg(
            {f: ['median'] for f in features}
        )
        df.columns = df.columns.droplevel(1)
        df[CLUSTER] = 0
        cluster_col_idx = df.columns.get_loc(CLUSTER)
        d = pdist(df[features])
        z = linkage(d, 'ward')
        threshold = (threshold or 0.7) * max(z[:, 2])
        selected = {}
        nb_leaf = len(df)
        for i in range(z.shape[0]):
            if z[i, 2] > threshold:
                res = get_all_child_from_both_side(i + nb_leaf, z,
                                                   nb_leaf, selected)
                assign_cluster_id = max(selected.values() or [0])
                if res[0]:
                    assign_cluster_id += 1
                    df.iloc[res[0], cluster_col_idx] = assign_cluster_id
                if res[1]:
                    assign_cluster_id += 1
                    df.iloc[res[1], cluster_col_idx] = assign_cluster_id

                selected[i] = assign_cluster_id
        return df

    result = hits.groupby(LIGNEE, sort=False).apply(by_lines,
                                                    threshold=color_threshold)
    return result[[CLUSTER]]


def is_deep_learning_feature(feature):
    """Return True if feature is a name
    for a deep learning feature else False

    Parameters
    ----------

    feature: str
        name of the feature

    """
    return str(feature).startswith(FEATURE_NAME_PREFIX_DEEP_LEARNING)


def separate_wavelength_features(features_list, fkeep=None):
    """
    create a dict for inference feature based on the wavelength (key)

    Parameters
    ----------

    features_list: list of str
        list of features used to create the dict

    fkeep: None or list of str
        name of groups (wavelenght) to keep in the resulting dict

    Returns
    -------

    result: dict of str
        the resulting dict
    """
    result = {}
    for f in features_list:
        if not is_deep_learning_feature(f):
            raise ValueError(f'{f} is not a valid feature for separating by wavelength')
        try:
            prefix, wave, *_ = f.split('_')
        except (TypeError, ValueError):
            raise ValueError(f'{f} is not a valid feature for separating by wavelength')
        if wave not in result:
            result[wave] = []
        result[wave].append(f)
    if fkeep is not None:
        result = {k: v for k, v in result.items() if k in fkeep}
    return result
