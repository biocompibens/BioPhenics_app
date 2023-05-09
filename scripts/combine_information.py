import os
from collections import OrderedDict
from io import BytesIO

from pandas import ExcelFile, read_excel, Series, concat, ExcelWriter, DataFrame
from sklearn.decomposition import PCA

from scripts.global_name import (REPLICAT, NOT_FEATURE, BARCODE, SIRNA_NAME,
                                 ALL_DATA, FIELDS, IMG_PATH, FIELDS_DATA,
                                 CONTENT, LIGNEE, FEATURE_NAME_PREFIX_DEEP_LEARNING,
                                 WELLS, PLATE, COL_ORDERED)
from scripts.fct_utils import separate_wavelength_features, form_well, clear_compound_name, sort_column, merge_rep


class SamePlateError(Exception):
    pass


class PreNormalization(object):
    """
    This class will take post-parsing bank, exp and raw files
    and merge them into one.
    The experience file will link data from raw and from bank.
    (the barcode in rawdata and the plate name in bank)

    Parameters
    ----------

    unified_file : A :class:`ExcelFile` or Path-like object, optional
        The file resulting of :func:`to_excel` of this class.
        Allow one to get an instance of :class:`PreNormalization`
        without redoing all the computation

    ctrl : List of String or String, default empty
        The name of the compound serving as control for the project
    """

    def __init__(self, unified_file=None, ctrl=None, not_selected=None):
        self.original_content = None
        self.book = {}
        self.deep_learning = False
        self.deep_features_list = None
        self.project_id = None
        self._log_df = None
        self.replicate_number = 0
        self.replicate_name = []

        if isinstance(ctrl, list):
            if all([isinstance(t, str) for t in ctrl]):
                self.ctrl = ctrl
            else:
                raise ValueError(f'{ctrl} is not a valid list of control')
        elif isinstance(ctrl, str):
            self.ctrl = [ctrl]
        else:
            self.ctrl = [""]

        self.fields = None

        self.img_path = None

        self.data = None

        try:
            is_file = os.path.isfile(unified_file)
        except (OSError, TypeError):
            is_file = False

        if (unified_file is not None and
                (isinstance(unified_file, (ExcelFile, BytesIO)) or is_file)):
            self.from_file(unified_file, not_selected=not_selected)

        elif self.data is not None:
            self.validate_data(not_selected=not_selected)

    def handle_multi_content(self):
        if CONTENT in self.data:
            col_condition = [c for c in self.data
                             if CONTENT in c and c != CONTENT]
            if col_condition:
                # reccord original columns and make an unique one
                # with all elem separate by ' + '
                self.original_content = self.data[[CONTENT] + col_condition]
                self.data[CONTENT] = self.data[CONTENT].str.cat(
                    self.data[col_condition], sep=" + "
                )
                self.data = self.data.drop(col_condition, axis=1)

    def count_rep(self):
        """Count the number of replicate present in project"""
        if self.data is not None:
            group = self.data.groupby([REPLICAT])
            self.replicate_number = len(group)

            self.replicate_name = group.groups

    def is_sirna(self):
        """ Return True if this experiment is a siRNA experiment or False
                otherwise """
        name_hits_count = self.data[CONTENT].value_counts()

        return name_hits_count.median() != self.replicate_number * len(self.data[LIGNEE].unique())

    def extract_wavelength(self):
        if self.deep_learning:
            return list(separate_wavelength_features(self.deep_features_list).keys())

    def features(self):
        """List all feature available"""
        return [f for f in self.data.columns if f not in NOT_FEATURE]

    def validate_data(self, not_selected=None):
        """
        Validate data from files
        From now it need :

        * All feature column must be castable into float (decimal number)
        * All meta-data column must be castable into str (string) except
          concentrations columns
        * If no :const:`~internal_script.global_name.BARCODE` column
          is defined it need to have one plate by (line, replicate)
          with unique wells
        * If no :const:`~internal_script.global_name.PLATE` column
          is defined, it need to have only one replicate and one line
        * If no :const:`~internal_script.global_name.REPLICATE` column
          is defined, it need to have one unique barcode by (plate, line)
        * If no :const:`~internal_script.global_name.LIGNEE` column
          is defined, it need to have one unique barcode by
          (plate, replicate)
        * If :const:`~internal_script.global_name.CONCENTRATION` column
          is defined, it need to not have
          :const:`~internal_script.global_name.CONCENTRATION_1` or
          :const:`~internal_script.global_name.CONCENTRATION_2` and the
          concentration values must be (at least one) differents from 0
        * If :const:`~internal_script.global_name.CONCENTRATION_1` or
          :const:`~internal_script.global_name.CONCENTRATION_2` is defined,
          the other must be defined too
        * If :const:`~internal_script.global_name.CONTENT` is not defined,
          so :const:`~internal_script.global_name.CONTENT_1` and
          :const:`~internal_script.global_name.CONTENT_2` must be defined
        * :const:`~internal_script.global_name.WELLS` must be defined


        """
        self.data = self.data.dropna(axis=0, how='all')
        self.data = self.data.dropna(axis=1, how='all')

        feature_col = [c for c in self.data if c not in NOT_FEATURE]
        other_col = [c for c in self.data if c in NOT_FEATURE]

        self.data = self.data[other_col + feature_col]

        if not_selected is not None:
            if (rep := not_selected.get('rep', False)) and REPLICAT in self.data:
                self.data = self.data.loc[~self.data[REPLICAT].isin(rep)].reset_index(drop=True).copy()
            if (lines := not_selected.get('lines', False)) and LIGNEE in self.data:
                self.data = self.data.loc[~self.data[LIGNEE].isin(lines)].reset_index(drop=True).copy()

        for col in feature_col:
            if self.data[col].isna().all():
                self.data = self.data.drop(columns=col)
                feature_col.pop(feature_col.index(col))
            else:
                try:
                    self.data[col] = self.data[col].astype(float)
                except ValueError:
                    feature_col.pop(feature_col.index(col))
                    try:
                        print(f"column {col} was removed")
                    except OSError:
                        pass
                except TypeError:
                    print(f"error with {col} on converting in float")
                    raise
            if not self.deep_learning:
                self.deep_learning = col.startswith(FEATURE_NAME_PREFIX_DEEP_LEARNING)

        self.data = self.data[other_col + feature_col]

        for col in other_col:
            self.data[col] = self.data[col].astype(str)
            if not self.data[col].sum():
                self.data = self.data.drop(columns=col)

            if self.img_path is not None and col in self.img_path:
                self.img_path[col] = self.img_path[col].astype(str)

        self.data[WELLS] = self.data[WELLS].apply(form_well)

        if BARCODE not in self.data.columns:
            if PLATE not in self.data.columns:
                raise ValueError(f'{PLATE} not found in data')
            grpby = [PLATE]

            if REPLICAT in self.data.columns:
                grpby.append(REPLICAT)

            if LIGNEE in self.data.columns:
                grpby.append(LIGNEE)

            for en, (p, df) in enumerate(self.data.groupby(grpby, sort=False)):
                if len(df[WELLS]) != len(df[WELLS].unique()):
                    raise ValueError('Non unique wells by plate')
                self.data.loc[df.index, BARCODE] = Series(en, index=df.index)

        if PLATE not in self.data.columns:
            if REPLICAT in self.data.columns:
                if len(self.data[REPLICAT].unique()) != 1:
                    raise ValueError(f'column {PLATE} missing')
            if LIGNEE in self.data.columns:
                if len(self.data[LIGNEE].unique()) != 1:
                    raise ValueError(f'column {PLATE} missing')
            self.data[PLATE] = self.data[BARCODE]

        if REPLICAT not in self.data.columns:
            grpby = [PLATE]
            if LIGNEE in self.data.columns:
                grpby.append(LIGNEE)

            for p, df in self.data.groupby(grpby, sort=False):
                if len(df[BARCODE].unique()) != 1:
                    raise ValueError(f'column {REPLICAT} missing')
            self.data[REPLICAT] = Series("R1", index=self.data.index)
        self.count_rep()

        if LIGNEE not in self.data.columns:
            for p, df in self.data.groupby([REPLICAT, PLATE], sort=False):
                if len(df[BARCODE].unique()) != 1:
                    raise ValueError(f'column {LIGNEE} missing')
            self.data[LIGNEE] = Series(" ", index=self.data.index)

        if CONTENT not in self.data.columns:
            raise KeyError("Wrong format for data, "
                           f"{CONTENT} not in columns names")
        col_content = [c for c in self.data.columns if CONTENT in c]
        self.data[col_content] = self.data[col_content].applymap(clear_compound_name)
        self.handle_multi_content()

        if WELLS not in self.data.columns:
            raise KeyError("Wrong format for data, "
                           f"{WELLS} not in columns names")

        nb_rep = len(self.data[REPLICAT].unique())

        if not all([len(df[REPLICAT].unique()) == nb_rep
                    for _, df in self.data.groupby(LIGNEE)]):
            raise ValueError("Lines don't have the same replicate name")

        if SIRNA_NAME in self.data.columns:
            self.data[SIRNA_NAME] = self.data[SIRNA_NAME].fillna("").astype(str)

        return 1

    def reduce_feature_space(self, n_components=0.99, svd_solver='full', by_wave=True):
        """"""
        if not self.deep_learning:
            raise ValueError('Can only reduce deep learning features')

        features = [col for col in self.data.columns if FEATURE_NAME_PREFIX_DEEP_LEARNING in col]
        if "PCA" in features[0]:
            # already done
            return

        pca = PCA(n_components=n_components, svd_solver=svd_solver)

        grp = separate_wavelength_features(features) if by_wave else {'': features}

        inter = []
        for wave, grp_features in grp.items():
            inter.append(DataFrame(pca.fit_transform(self.data[grp_features].to_numpy()), index=self.data.index))
            inter[-1].columns = [f'{FEATURE_NAME_PREFIX_DEEP_LEARNING}{wave + "_" if wave else ""}PCA_{i}'
                                 for i in range(len(inter[-1].columns))]
        result = concat(inter, axis=1)
        self.data = concat([self.data[[c for c in self.data if c not in features]], result], axis=1)

    def get_original_content(self, df=None):
        """
        replace modified value by their original ones

        Parameters
        ----------
        df: DataFrame
            actual data

        Returns
        -------
        original data

        """
        if df is None:
            df = self.data
        df = df.copy()
        if self.original_content is not None:
            df[self.original_content.columns] = self.original_content
        return df

    # ### output ####

    def to_excel(self, out, pageby=REPLICAT, mergeby='',
                 my_sheet=ALL_DATA, personalise=True, overwrite=True):
        """
        write an excel file 'out', each value for 'foreach'
        ("wells" or "fields"), create a page by 'pageby' and merge
        data according to 'mergeby'. if personalise=False,
        these parameters are not taken into account

        Parameters
        ----------
        overwrite: Boolean
            if true overwrite the file "out" otherwise append data to it
            (if possible)
        out: string,
            Path of the output file
        pageby: String
            Name of a column of self.data, a spreadsheet is created
            by different values of this column
        mergeby: String
            Name of a column of self.data, merge the data
            by each value of this column
        my_sheet: String,
            Name of the spreadsheet if pageby = ''
        personalise: boolean,
            Reccord data in a cusom format, if false some of
            the previous parameters are not taken into account

        Returns
        -------
        None

        """
        if not overwrite:
            try:
                self.book = read_excel(out, sheet_name=None, engine="openpyxl")
            except (IOError, TypeError):
                self.book = {}
        else:
            self.book = {}

        if not personalise or (not pageby and not mergeby):  # Format classique d'enregistrement
            self.book[ALL_DATA] = self.get_original_content()

        else:  # format personnalisable mais non relu
            my_input = self.get_original_content(self.data)
            if my_input is not None:
                if pageby:
                    if pageby not in my_input.columns:
                        raise KeyError(f"{pageby} not found in data")

                    for i, (line, donnee) in enumerate(
                            my_input.groupby(pageby, sort=False)
                    ):
                        donnee = sort_column(
                            donnee.dropna(how='all', axis=1)
                        ).drop(pageby, axis=1)

                        if mergeby and mergeby != pageby:
                            col_to_keep = [col for col in NOT_FEATURE
                                           if col in donnee.columns
                                           and col != mergeby]
                            if mergeby == REPLICAT:
                                col_to_keep.remove(BARCODE)
                            donnee = merge_rep(donnee, mergeby, col_to_keep)

                        self.book[line] = donnee

                else:  # pas de pageby
                    donnee = sort_column(my_input.dropna(how='all', axis=1))
                    if mergeby:
                        col_to_keep = [col for col in
                                       COL_ORDERED + [FIELDS] if
                                       col in donnee.columns and col != mergeby]
                        if mergeby == REPLICAT:
                            col_to_keep.remove(BARCODE)
                        donnee = merge_rep(donnee, mergeby, col_to_keep)

                    self.book[my_sheet] = donnee
            else:
                return

        if self.img_path is not None and not self.img_path.empty:
            self.book[IMG_PATH] = self.img_path
        field = self.fields
        if field is not None:
            self.book[FIELDS_DATA] = field
        u = 1
        files = out if isinstance(out, ExcelWriter) else ExcelWriter(out)
        with files as xlsx:
            for sheet, df in self.book.items():
                if sheet is None:
                    df.to_excel(xlsx, f"other_{u}", index=False,
                                engine="openpyxl")
                    u += 1
                else:
                    try:
                        df.to_excel(xlsx, str(sheet), index=False,
                                    engine="openpyxl")
                    except AttributeError:
                        pass

    def _from_custom_file(self, donnee, sheetname, page_by, merge_by, not_selected):
        """
        Take a customized file (from self.to_excel with personalise=True)
        and try to convert it into an instance of this classe.

        Parameters
        ----------
        donnee: String or ExcelFile
            the pathname of the file or an instance of ExcelFile to be read.
        page_by: String
            The name of the column used to created the page
        merge_by: String
            The name of the column used to merged data

        Returns
        -------
        PreNormalization instance
            The newly created instance of prenormalization from the file.

        """

        result = OrderedDict()
        result[ALL_DATA] = DataFrame()

        if page_by is not None:
            for name, df in donnee.items():
                if name not in [sheetname[1], sheetname[2]]:
                    result_df = concat([Series(name,
                                               name=page_by,
                                               index=df.index), df],
                                       axis=1)
                    result[ALL_DATA] = concat([result[ALL_DATA],
                                               result_df],
                                              axis=0, ignore_index=True)
                else:
                    result[name] = df
        else:
            result[ALL_DATA] = donnee[sheetname[0]]
            if sheetname[1] in donnee:
                result[FIELDS_DATA] = donnee[sheetname[1]]
            if sheetname[2] in donnee:
                result[IMG_PATH] = donnee[sheetname[2]]

        if merge_by is not None:
            col_kept = []
            new_df = None
            for col in result[ALL_DATA].columns:
                try:
                    col_name, merge_name = col.rsplit("_", 1)
                except ValueError:
                    col_kept.append(col)
                    continue

                if new_df is None:
                    new_df = DataFrame({merge_by: merge_name,
                                        col_name: result[ALL_DATA][col]})
                elif merge_name in new_df[merge_by].values:
                    e = result[ALL_DATA][col]
                    if col_name in new_df.columns:
                        e.index = new_df.loc[new_df[merge_by] == merge_name,
                                             col_name].index
                        new_df[col_name] = new_df[col_name].fillna(e)
                    else:
                        new_df[col_name] = e
                else:
                    new_df = concat(
                        [new_df,
                         DataFrame({merge_by: merge_name,
                                    col_name: result[ALL_DATA][col]})],
                        axis=0, ignore_index=True, sort=False
                    )
            inter = concat(
                [result[ALL_DATA][col_kept]] * len(new_df[merge_by].unique()),
                axis=0
            )
            inter = inter.reset_index(drop=True)
            result[ALL_DATA] = concat([inter, new_df], axis=1)

        return self._from_file(result, [ALL_DATA, FIELDS_DATA, IMG_PATH], not_selected=not_selected)

    def _from_file(self, data, sheetnames, not_selected):
        """
        Take a non-customized file (from self.to_excel with personalise=False)
        and reccord it into self.

        Parameters
        ----------
        data: Pathname or ExcelFile instance
            The name of the file to be read.
        """
        try:
            self.data = data.pop(sheetnames[0])
        except KeyError:
            raise KeyError("Wrong sheetnames, "
                           f"'{sheetnames[0]}' is not in the file")
        self.fields = data.pop(sheetnames[1], None)
        self.img_path = data.pop(sheetnames[2], None)
        self.validate_data(not_selected=not_selected)

    def from_file(self, the_file, sheetnames=None, not_selected=None,
                  col_name=None, pageby=None, merge_by=None):
        """
        Clean input for the parsing (:func:`self._from_file`)
        Determine the column in the_file to get the correct parsing function

        Parameters
        ----------
        the_file: OrderedDict or DataFrame or BytesIO
            pandas representation of multiple sheet of the excel file

        sheetnames: List of string
            the name of the different sheet to be parsed
            (default : [:const:`~internal_script.global_name.ALL_DATA`,
            :const:`~internal_script.global_name.FIELDS_DATA`,
            :const:`~internal_script.global_name.IMG_PATH`])

        not_selected: dict
            dict with at most two keys : "rep" and / or "lines"
            with values a list of replicate and/or lines to remove from analysis

        col_name: dict of mapped column
            dict to rename the dataframe columns (key: column in the file,
            value: same as defined in :class:`~internal_script.global_name`

        pageby: String
            Name of the column used to create each sheet

        merge_by: String
            Name of the column used to merged data

        Returns
        -------
        None

        """
        sheetnames = sheetnames if sheetnames is not None else [
            ALL_DATA, FIELDS_DATA, IMG_PATH
        ]
        col_name = col_name if col_name is not None else {}
        if not isinstance(sheetnames, list):
            if isinstance(sheetnames, str):
                if len(sheetnames.split(" ")) > 1:
                    sheetnames = sheetnames.split(" ")
                else:
                    sheetnames = [sheetnames, FIELDS_DATA, IMG_PATH]
            else:
                raise KeyError("Wrong sheetnames")
        else:
            if len(sheetnames) != 3:
                raise KeyError("Wrong formatting for sheetnames")

        if not isinstance(the_file, (OrderedDict, dict)):
            try:
                donnee = read_excel(the_file, sheet_name=None)
            except IOError as err:
                raise IOError(f"Error while reading {the_file} : {err}")
        else:
            donnee = the_file

        for s in donnee.keys():
            if donnee[s] is not None:
                donnee[s] = donnee[s].rename(col_name, axis=1)
                # rename colnames for each sheet

        if merge_by is None and pageby is None:
            return self._from_file(donnee, sheetnames, not_selected)
        else:
            return self._from_custom_file(donnee, sheetnames, pageby, merge_by, not_selected)
