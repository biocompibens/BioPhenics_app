import json

from numpy.linalg import LinAlgError, cond, pinv
from scipy.linalg import inv
import numpy as np
from scipy.ndimage import generic_filter1d, median_filter
from pandas import concat, DataFrame, MultiIndex, Series
from scipy.spatial.distance import mahalanobis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from scripts.global_name import (
    LABEL, CONTENT, MAHALANOBIS, BARCODE, LDA,
    WELLS, LIGNEE, PLATE, ROW, COLUMN, REPLICAT, LDA_COL, LR_COLNAME, CELL_COUNT, LUMINESCENCE,
    INFERENCE_WAVE_BOOL)

from scripts.fct_utils import plate_and_back, get_commun_ctrl, \
    extract_row_and_col, separate_wavelength_features, join_and

methods = {}


class _MetaMethod(type):

    def __init__(cls, nom, bases, dic):
        type.__init__(cls, nom, bases, dic)
        try:
            methods[getattr(cls, 'step')] = cls
            # register each classes in methods dictionary
        except AttributeError:
            pass


class Method(metaclass=_MetaMethod):
    """
    Class to be subclasses by all categories of methods used
    in the differents steps of the normalization process
    """
    step: str

    def __new__(cls, step='', *args, **kwargs):
        cls = methods.get(step, cls)
        # allow to initialize the correct class with step parameter
        # ex = Method(step="transformation") will initialize
        # a TransformationMethod instance
        return object.__new__(cls)

    def _get_function(self, attr_str):
        """ """
        if not attr_str:
            return

        func_name = attr_str.replace(" ", "_")
        try:
            func = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError(f"{attr_str} is not a valid method")
        return func

    def _record_parms(self, name, kwargs, doc):
        """reccord name, parms and reference for the function used"""
        self.name = name
        self.parms = kwargs
        try:
            self.ref = doc.split('`')[-2]
        except IndexError:
            pass

    def __call__(self, data, parms):
        """
        Make the class callable, perform the recording
        of parms and find the function (need to be subclassed in order
        to perform the actual computation)
        """
        methode = parms.get(self.step)
        func = self._get_function(methode)
        doc = func.__doc__ or '' if func is not None else ''
        kwargs = parms.get_parms_of(self.step)
        kwargs.update(**parms.other_parms)
        self._record_parms(methode, kwargs, doc)
        return func

    @classmethod
    def get_methods(cls):
        """Get the list of the method available for the class"""
        return [method.replace('_', ' ') for method in cls.__dict__.keys()
                if not method.startswith("_") and method not in
                ["get_methods", "get_parameters", "get_info", "step"]]

    @classmethod
    def get_parameters(cls, methode):
        """Get the parameters of the method and their default value"""
        if not methode:
            return

        methode = methode.replace(" ", "_")
        try:
            do_method = getattr(cls, methode)
        except AttributeError:
            raise NotImplementedError(f"{methode} is not a valid method")
        except TypeError:
            return

        try:
            this_code = do_method.__code__
            this_default = do_method.__defaults__
            result = {
                cle: val for cle, val in
                zip(this_code.co_varnames[:this_code.co_argcount][::-1],
                    this_default[::-1])
            }
            return {r: result[r] for r in reversed(list(result))}
        except TypeError:
            return None


class TransformationMethod(object):
    """Method for the numerical transformation"""
    step = "transformation"

    @staticmethod
    def logit(v):
        """
        Transform a distribution between 0 and 1 (or 0 and 100) into a distribution
        in :math:`\\mathbb{R}`, \n
        :math:`logit = \\frac{log(x)}{log(1-x)}`
        """
        max_v = v.max()
        if max_v > 1:
            if max_v > 100:
                raise ValueError("Values are not between 0 and 1 (or 0 and 100)")
            v = v / 100
        if v.min() < 0:
            raise ValueError('There are negatives values, logit invalid')

        v[v <= 0] = 1e-12
        v[v >= 1] = 1 - 1e-12
        return np.log(v) - np.log(1.0 - v)

    @staticmethod
    def inv_logit(x):
        return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def log(x):
        """
        Compute the *natural logarithm* of x.
        """
        return np.log(x)

    @staticmethod
    def inv_log(x):
        """
        inverse of log
        :param x:
        :return:
        """
        return np.exp(x)


class CorrectionMethod(Method):
    """Methods for spatial normalization"""
    step = "spatial_correction"

    @classmethod
    def get_parameters(cls, methode):
        result = super().get_parameters(methode)
        if result is not None:
            result['log'] = []
        # add a supplementary parameters as it is not used
        # in methode directly but in the class body
        return result

    def __call__(self, data, parms):
        do_method = super().__call__(data, parms)
        data = data.copy()
        # doesnt back-propagate change
        if do_method is None:
            return

        transform_recap = {}
        percent_recap = []
        for f in parms.features:
            if f in [CELL_COUNT, LUMINESCENCE]:
                transform_recap[f] = TransformationMethod.inv_log
                data[f] = TransformationMethod.log(data[f].copy())
                continue
            if 0 <= data[f].min() and data[f].max() <= 100:
                percent_recap.append(f)
                data[f] = data[f] / 100

            if 0 <= data[f].min() and data[f].max() <= 1:
                transform_recap[f] = TransformationMethod.inv_logit
                data[f] = TransformationMethod.logit(data[f].copy())

        result = do_method(data, parms.features, **self.parms, ctrl=parms.ctrl).copy()

        for f, func in transform_recap.items():
            result[f] = func(result[f])

        for f in percent_recap:
            result[f] = result[f] * 100

        non_negative_features = [CELL_COUNT, LUMINESCENCE]

        for nnf in non_negative_features:
            if nnf in parms.features:
                result.loc[result[nnf] < 0, nnf] = 0

        return result

    @staticmethod
    def median_polish_thouis(df, feature, ctrl, iteration=10, align='Once',
                             align_method='All But Controls', **_):
        """
        Same algorithm than thouis one
        (compute the median of values in same row, col from all plates
        whithin a replicate, then substract the median of these medians to it.
        the resulting values is obtained by substract this to the original
        value.)
        """
        def _align_thouis(data):
            if align_method == 'All But Controls':
                align_values = data.loc[~data[CONTENT].isin(ctrl)]
            elif align_method == 'All Wells':
                align_values = data
            else:
                raise ValueError(f'wrong align method : {align_method}')

            offsets = align_values.groupby(BARCODE)[feature].agg("median")
            # median by plates

            rep = data.groupby(BARCODE)[REPLICAT].agg('first')
            offsets = offsets.groupby(rep, group_keys=False).apply(
                lambda x: x - x.median()
            )
            # median by plate - median by replicate

            return data.groupby(BARCODE, group_keys=False)[feature].apply(
                lambda x: x - offsets.loc[x.name]
            )

        def _fix_nans2(shift_vals):
            cols = shift_vals.columns
            index_name = shift_vals.index.name or 0
            tmp_df = shift_vals.sort_index().reset_index()
            tmp_df[cols] = tmp_df[cols].interpolate(method='nearest').fillna(method="ffill").fillna(method="bfill")
            # fill first and last nan if there is ... idk how to make nearest work for those
            return tmp_df.set_index(index_name)

        def _conservative_nanmedian2(arr):
            if arr.size == 0:
                return
            r = arr.median()
            r[arr.isna().sum() * 2 > len(arr)] = np.nan
            return r

        def _apply_median_polish(data):
            # remove ctrl
            control_values = data.loc[res[CONTENT].isin(ctrl), feature]
            data.loc[res[CONTENT].isin(ctrl), feature] = np.nan

            # calculate offset by row / col
            offset = _fix_nans2(data[feature].groupby(
                row_col[direction]
            ).agg(_conservative_nanmedian2))
            offset -= offset.median()

            # replace value in ctrl
            data.loc[res[CONTENT].isin(ctrl), feature] = control_values

            # apply offset on values
            data.loc[:, feature] = data[feature].groupby(
                row_col[direction], group_keys=False
            ).apply(lambda x: x - offset.loc[x.name])
            return data

        result = DataFrame()
        for _, ddf in df.groupby(LIGNEE):
            row_col = extract_row_and_col(ddf[WELLS])
            res = ddf.copy()
            for i in range(iteration):
                if (i == 0 and align == "Once") or (align == 'Each iteration'):
                    res[feature] = _align_thouis(res)

                for direction in (ROW, COLUMN):
                    res = res.groupby(REPLICAT, group_keys=False).apply(_apply_median_polish)

            result = concat([result, res], axis=0)
        return result[feature]


class NormalisationMethod(Method):
    """Methods for Plate alignment process"""
    step = "normalization"

    def __call__(self, data, parms):
        do_method = super().__call__(data, parms)
        if do_method is None:
            return
        result = do_method(data, feature=parms.features, ctrl_neg=parms.ctrl_neg,
                                   ctrl_pos=parms.ctrl_pos, **self.parms)
        return result

    @staticmethod
    def _MAD(serie):
        """MAD: :math:`median( \\lvert Y_i – median(Y_i) \\rvert )`"""
        return (abs(serie - serie.median())).median()

    @staticmethod
    def _alignment_by_plate(df, func):
        result = df.groupby(BARCODE, sort=False, group_keys=False).apply(func)
        if isinstance(result.index, MultiIndex):
            result = result.reset_index(level=0, drop=True)
        return result

    @staticmethod
    def _get_population(df, population, ctrl_neg):
        if population == "ctrl":
            condition = df[CONTENT].isin(ctrl_neg)
        elif population == 'common':
            common_ctrl = get_commun_ctrl(df, ctrl_neg)
            if common_ctrl.empty:
                raise ValueError('No common control '
                                 'for all plates can be found')
            condition = Series(df.index.isin(common_ctrl.index), index=df.index)
        else:  # sample
            condition = Series(True, index=df.index)
        return condition

    def robust_z_score(self, df, feature,
                       ctrl_neg, population='ctrl', **_):
        """
        For each plate we have :
        :math:`robust\\ Z\\ score = \\frac{X_i - median_{pop}}{MAD_{pop} * 1.4826}`

        with :math:`MAD = median(| X_i - median(X_i)|)`

        `Brideau, C., Gunter, B., Pikounis, B., & Liaw, A. (2003).
        Improved statistical methods for hit selection
        in high-throughput screening.
        Journal of biomolecular screening, 8(6), 634-647.`
         """

        condition = self._get_population(df, population, ctrl_neg)

        def f(x):
            pdf = x.loc[condition, feature]
            try:
                return (x[feature] - pdf.median()) / (self._MAD(pdf) * 1.4826)
            except ZeroDivisionError:
                return np.nan

        return self._alignment_by_plate(df, f)


class HitSelection(Method):
    """Method for selection criteria"""
    step = "selection"

    def __call__(self, data, parms):
        """Main change with the parent class is that
        it dont concat the result (the feature are evaluated together)"""
        do_method = super().__call__(data, parms)
        if do_method is None:
            return

        result = do_method(data, features=parms.features,
                           ctrl_neg=parms.ctrl_neg,
                           ctrl_pos=parms.ctrl_pos, **self.parms)

        return result

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.additional_features = None

    @staticmethod
    def _separate_wavelength_features(features, str_parms, inference_wavelength_bool):

        if inference_wavelength_bool and inference_wavelength_bool != 'no':
            grp_features = separate_wavelength_features(
                features,
                fkeep=[rule['feature'].split('_')[0] for rule in str_parms]
            )
        else:
            grp_features = {'': features}
        return grp_features

    @staticmethod
    def _selection_by_constant(df, feature_to_compute, threshold, op):
        """Do the selection rules by rules

        Parameters
        ----------

        df: pandas DataFrame
            the data on which the selection is performed
        feature_to_compute: String
            the column name of the data to be considered for this rule
        threshold: number
            the value by which the data are selected
        op: String
            can be '<', '>', or '><' to select all data that are lower, greater
            or their absolute value greater than the threshold respectively

        Returns
        -------
        The result of the selection (mask)
        """
        try:
            threshold = float(threshold)
        except TypeError:
            raise TypeError('Invalid value for threshold in hit selection')

        def f(x):
            pass

        if op == '<':
            def f(x):
                return x < threshold
        elif op == '>':
            def f(x):
                return x > threshold
        elif op == '><':
            def f(x):
                return abs(x) > abs(threshold)
        else:
            if op != 'b' and op != 'w':
                raise ValueError(f"Invalid operator : '{op}'")

        groups = [LIGNEE, PLATE, WELLS]

        valid_rep = df.groupby(
            groups, sort=False
        )[feature_to_compute].agg('median')

        if op in 'bw':
            threshold = int(threshold)
            sorted_ = valid_rep.sort_values()
            index_ = sorted_.iloc[:threshold].index if op == "w" else sorted_.iloc[-threshold:].index
            valid_rep = valid_rep.to_frame()
            valid_rep.loc[index_, 'bool'] = True
            return df[groups].merge(valid_rep['bool'].fillna(False).reset_index(), on=groups, how='left')['bool']

        valid_rep = valid_rep.apply(f).reset_index()
        valid_rep = df[groups].merge(valid_rep, on=groups, how='left')
        # make the test on the median of the replicat as well (in certain cases,
        # it can remove some false hit)
        return (df[feature_to_compute].apply(f) & valid_rep[feature_to_compute]).fillna(False)

    def selection_by_constant(self, df, features, ctrl_neg,
                              ctrl_pos, str_parms, **_):
        """Perform a hit selection given the rule below"""
        result = df.copy()
        inter = [(p["include"],
                  self._selection_by_constant(df, p['feature'],
                                              p["value"], p["relative"]))
                 for p in str_parms]
        res = inter.pop(0)[1]

        for include, p in inter:
            if include == 'and':
                res &= p
            elif include == 'or':
                res |= p

        result[LABEL] = res

        ctrl_hit = result[CONTENT].isin(ctrl_neg)
        # remove hits that are control
        if ctrl_pos is not None:
            ctrl_hit = ctrl_hit | result[CONTENT].isin(ctrl_pos)

        result.loc[ctrl_hit, LABEL] = False

        return result

    def Mahalanobis_distance(self, df, features, ctrl_neg,
                             ctrl_pos, str_parms, **kwargs):
        """
        Compute the distance between a sample and the mean
        of control in the feature space. If a positive control is given,
        the distance is computed by their distribution else
        it is by the negative control distribution.
        (work on one-feature analysis but it's better suited
        for a multivariate analysis)

        The distance is given by :

        :math:`D_M (x) = \\sqrt{(x - \mu)^T S^{-1} (x - \mu)}`,

        with S = control distribution covariance matrix, x = sample vector
        and :math:`\mu` = control distribution mean vector

        Ref : `Mahalanobis, P. C. (1936).
        "On the generalized distance in statistics".
        National Institute of Science of India.`
        """
        result, features = self._clean_na(df, features, 0)

        if ctrl_pos:
            ctrl = result.loc[result[CONTENT].isin(ctrl_pos), features]
        else:
            ctrl = result.loc[result[CONTENT].isin(ctrl_neg), features]

        if ctrl.empty:
            raise ValueError('No control population for a plate')

        grp_features = self._separate_wavelength_features(
            features, str_parms,
            kwargs.get('other_parms', {}).get(INFERENCE_WAVE_BOOL, False)
        )

        for w, f in grp_features.items():
            try:
                cov_inv = inv(ctrl[f].cov().values)
            except LinAlgError:
                err_msg = 'Can not compute Mahalanobis distance on '
                if w:
                    err_msg += f' "{w}" wavelength features.'
                else:
                    if len(f) > 4:
                        err_msg += 'those features.'
                    else:
                         err_msg += join_and(f) + '.'
                raise ValueError(err_msg)

            result[(w + '_' if w else '') + MAHALANOBIS] = result[f].apply(
                mahalanobis, v=ctrl[f].mean(axis=0), VI=cov_inv,
                axis=1
            )
        maha_feat = [f for f in result.columns if MAHALANOBIS in f]
        if result[maha_feat].isna().all().any():
            raise ValueError('Can not compute Mahalanobis distance. '
                             'You need to choose another method')

        self.additional_features = result[maha_feat]
        return self.selection_by_constant(result, features, ctrl_neg,
                                          ctrl_pos, str_parms)

    @staticmethod
    def _clean_na(df, col_name, na_tol=383):  # if a plate equivalent of na
        df[col_name] = df[col_name].replace({np.inf: np.nan, -np.inf: np.nan})
        col_to_drop = df[col_name].columns[(df[col_name].isna().sum() > na_tol) | (df[col_name].std() == 0)]
        features = [f for f in col_name if f not in col_to_drop]
        return df.drop(columns=col_to_drop), features

    def linear_discriminant_analysis(self, df, features,
                                     ctrl_neg, ctrl_pos,
                                     str_parms, **kwargs):
        """
        Perform a linear discriminant analysis
        to predict the classes of the data.
        There is two class defined by negative and positive control.
        Therefore a positive control is required !

        Ref : `“The Elements of Statistical Learning”,
        Hastie T., Tibshirani R., Friedman J.,
        Section 4.3, p.106-119, 2008.`

        """
        if ctrl_pos is None:
            raise ValueError("You can not use a linear discriminant "
                             "analysis without a positive control")
        # clean data
        result, features = self._clean_na(df, features, 0)

        grp_features = self._separate_wavelength_features(
            features, str_parms,
            kwargs.get('other_parms', {}).get(INFERENCE_WAVE_BOOL, False)
        )
        n_components = 1

        df_ctrl_neg = df.loc[df[CONTENT].isin(ctrl_neg), features]
        label = np.array([0] * len(df_ctrl_neg))

        df_ctrl_pos = df.loc[df[CONTENT].isin(ctrl_pos), features]

        if df_ctrl_pos.empty:
            raise ValueError("positive control not found in data")
        label = np.concatenate([label, np.array([1] * len(df_ctrl_pos))])

        training_set = concat([df_ctrl_neg, df_ctrl_pos], axis=0)

        try:
            ctrl_pos2 = kwargs['other_parms']['ctrl_pos2']
            ctrl_pos3 = kwargs['other_parms'].get('ctrl_pos3', [])
        except (TypeError, KeyError):
            ctrl_pos2 = None
            ctrl_pos3 = None

        if ctrl_pos2:
            if len(features) < (2 if not ctrl_pos3 else 3):
                raise ValueError('Can not do LDA '
                                 'with more classes than feature selected')

            def _add_one_class(compound_list, training, current_label, nbc):
                if any(c in ctrl_pos + ctrl_neg for c in compound_list):
                    raise ValueError('Same compound used in different groups of '
                                     'control. This is not permitted.')
                df_to_add = df.loc[df[CONTENT].isin(compound_list)]
                current_label = np.concatenate([
                    current_label,
                    np.array([nbc] * len(df_to_add))
                ])
                training = concat(
                    [training, df_to_add], axis=0
                )
                return training, current_label

            n_components += 1

            training_set, label = _add_one_class(ctrl_pos2,
                                                 training_set, label, 2)

            if ctrl_pos3:
                n_components += 1
                training_set, label = _add_one_class(ctrl_pos3,
                                                     training_set, label, 3)

        for w, f in grp_features.items():
            clf = LinearDiscriminantAnalysis(n_components=n_components)
            clf.fit(training_set[f].values, label)
            cols_names = [(w + '_' if w else '') + LDA_COL[i] for i in range(n_components)]
            result[cols_names] = DataFrame(clf.transform(df[f].values))
            mean_ctrl_neg = result.loc[result[CONTENT].isin(ctrl_neg), cols_names].mean()
            for j, c in enumerate(cols_names):
                if j == 0:
                    pos = ctrl_pos
                elif j == 1:
                    pos = ctrl_pos2
                else:
                    pos = ctrl_pos3

                mean_ctrl_pos = result.loc[result[CONTENT].isin(pos), c].mean()
                if mean_ctrl_neg[c] > mean_ctrl_pos:
                    result[c] = - result[c]
        self.additional_features = result[[f for f in result.columns if LDA in f]]
        return self.selection_by_constant(result, LDA, df_ctrl_neg,
                                          df_ctrl_pos, str_parms, **kwargs)

    def _sirna_selection(self, df, features, ctrl_neg,
                        ctrl_pos, str_parms, **kwargs):
        """
        Select and pool SiRNA targeting the same gene (only available for SiRNA
        experiments). You have to select how many SiRNA need to be considered as
        hits for the gene to be as wells and the method to aggregate numerical
        data for the gene (default = 2nd strongest of SiRNA values)
        """
        result = self.selection_by_constant(df, features, ctrl_neg, ctrl_pos,
                                            str_parms, **kwargs)
        # do the pooling
        return result
