import ast
import copy
import json
import os

from numpy import inf

from scripts.global_name import DEEP_LEARNING_FEATURES, SHEETNAMES
from scripts.interval import Interval, AtomicInterval


def setter_parms(obj_type):
    class SetterParms(object):
        def __init__(self, func, doc=None):
            self.func = func
            self.__doc__ = doc if doc is not None else func.__doc__

        def __set__(self, obj, value):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except BaseException:
                    pass
            if not isinstance(value, obj_type) and value is not None:
                raise ValueError(f"Wrong type for {value} ({type(value)}), "
                                 f"need {obj_type}")
            if value is None:
                # default value
                value = obj_type()
            return self.func(obj, value)
    return SetterParms


class Parameters(object):
    """
    Dataclass that register all necessary parameters in order
    to run a normalisation
    """

    def __init__(self, json_string=None, **kwargs):
        dict_parms = json.loads(json_string) if json_string is not None else kwargs
        if isinstance(dict_parms, dict):
            dict_parms = [dict_parms]
        self._dict = {}
        self.filename = ''
        self.contents = ''
        self.other_parms = {}
        self._analysis = []
        self.not_selected = {}

        for analysis in dict_parms:
            if 'contents' in analysis:
                if not self.contents:
                    self.contents = analysis.pop('contents')
                    self.filename = analysis.pop('filename', '')
                else:
                    analysis.pop('contents')
                    analysis.pop('filename')
            if 'other_parms' in analysis:
                self.other_parms.update(analysis['other_parms'])
            if 'not_selected' in analysis:
                self.not_selected.update(analysis.pop('not_selected'))
            a = OneFeatParms(parent=self, **analysis)
            self._analysis.append(a)
            if 'features' in analysis:
                for f in analysis['features']:
                    self._dict[f] = a

    def __getitem__(self, item):
        if item in self._dict:
            return self._dict[item]
        for p in self._analysis:
            for rule in p.s_parms.get('str_parms', []):
                if item == rule['feature']:
                    return p
        raise KeyError(f'{item} not found in this parameters instance')

    def __getattr__(self, item):
        if item.startswith('_'):
            # redirect to __get_attribute__ when necessary
            raise AttributeError()
        result = list(getattr(v, item) for v in self._analysis)
        try:
            result = list(set(result))
        except TypeError:
            pass
        if len(result) == 1:
            result = result[0]
        elif result and isinstance(result[0], list):
            result = list(set(sum(result, [])))
            # flatened
        return result

    def __setattr__(self, key, value):
        # affect all analysis (you need to select a specified analysis if
        # you need more grained control)
        if ('_analysis' in self.__dict__
                and self._analysis
                and key in self._analysis[0].__dict__):

            for analysis in self:
                analysis.__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def __iter__(self):
        yield from self._analysis

    def __repr__(self):
        return ','.join(str(v) or 'None' for k, v in self.__dict__.items()
                        if not k.startswith('_')) + str(self._analysis)

    def __str__(self):
        return 'Parameters(' + os.linesep + os.linesep.join(
            [p.dumps() for p in self]
        ) + f"{os.linesep}filename='{self.filename}'{os.linesep})"

    def __len__(self):
        return len(self._analysis)

    @setter_parms(dict)
    def other_parms(self, value):
        self.__dict__["other_parms"] = value

    @property
    def s_parms_mixed(self):
        if isinstance(self.s_parms, dict):
            return self.s_parms
        for list_condition in self.s_parms:
            if list_condition and 'str_parms' not in list_condition:
                # in that particular cases we do nothing
                return self.s_parms
        else:
            result = [it for analysis in self.s_parms
                      for it in analysis.get('str_parms', {})]
            done_first = False
            for r in result:
                if 'include' in r and r['include'] is None:
                    if done_first:
                        r['include'] = 'and'
                    else:
                        done_first = True
            return {'str_parms': result}

    def get(self, name, defaults=None):
        try:
            return self.__getattr__(name)
        except KeyError:
            return defaults

    def is_norm_ready(self, content=True):
        if content and not self.contents:
            return False
        for analysis in self:
            if not analysis.features or not analysis.ctrl_neg:
                return False
        else:
            return True

    def get_parms_of(self, method_type):
        if method_type == "transformation":
            return getattr(self, "t_parms")
        elif method_type == "spatial_correction":
            return getattr(self, "sc_parms")
        elif method_type == "normalization":
            return getattr(self, "n_parms")
        elif method_type == "selection":
            return getattr(self, "s_parms")

    @property
    def ctrl(self):
        # if not get_attribute change the order of controls !
        return self.ctrl_neg + self.ctrl_pos

    def get_interval_of(self, feature):
        for a in self:
            possible_feat = [r['feature'] for r in a.s_parms.get("str_parms")]
            if feature in possible_feat:
                return a.get_interval_of(feature)

    @classmethod
    def from_excel(cls, file_name, sheet_name=SHEETNAMES[4]):
        import pandas as pd
        with pd.ExcelFile(file_name) as file_obj:
            df = pd.read_excel(file_obj, sheet_name=sheet_name)
        list_parms = df.dropna(subset=['Parameters :']).to_dict('split')['data']
        result = cls()
        result._analysis = []
        corresponding = {
            "Negative control": 'ctrl_neg', 'Positive control': 'ctrl_pos',
            'Transformation': ['transformation', 't_parms'],
            'Spatial normalization': ['spatial_correction', 'sc_parms'],
            'Plates alignment': ['normalization', 'n_parms'],
            'Hit selection': ['selection', 's_parms'],
            'Replicate hit validity': 'validation'
        }
        current_analysis = None
        for row in list_parms:
            name = row.pop(0)
            row = [r for r in row if not pd.isna(r)]
            if name == 'Feature analysed':
                if current_analysis is not None:
                    result._analysis.append(current_analysis)
                current_analysis = OneFeatParms()
                current_analysis.features = row
            elif name in corresponding:
                if current_analysis is None:
                    raise ValueError('Something went wrong')
                elif isinstance(corresponding[name], str):
                    if corresponding[name] == 'validation':
                        row = int(row[0][0])
                    current_analysis[corresponding[name]] = row
                elif row:
                    current_analysis[corresponding[name][0]] = row.pop(0)
                    parms_dict = {}
                    for cell in row:
                        parms_name, parms_value = cell.split(' : ', 1)
                        try:
                            parms_value = int(parms_value)
                        except ValueError:
                            try:
                                parms_value = float(parms_value)
                            except ValueError:
                                pass
                        parms_dict[parms_name] = parms_value

                    current_analysis[corresponding[name][1]] = parms_dict
                    if 'str_parms' in current_analysis[corresponding[name][1]]:
                        # json.loads will not work here as we took str(list(dict())) to write it on excel
                        current_analysis[corresponding[name][1]]['str_parms'] = ast.literal_eval(
                            current_analysis[corresponding[name][1]]['str_parms']
                        )
        result._analysis.append(current_analysis)
        return result

    def copy(self):
        return copy.deepcopy(self)


class OneFeatParms(object):
    @setter_parms(list)
    def ctrl_neg(self, value):
        self.__dict__["ctrl_neg"] = value

    @setter_parms(list)
    def ctrl_pos(self, value):
        self.__dict__["ctrl_pos"] = value

    @setter_parms(list)
    def ctrl_other(self, value):
        self.__dict__["ctrl_other"] = value

    @setter_parms(list)
    def features(self, value):
        self.__dict__["features"] = value

    @setter_parms(str)
    def transformation(self, value):
        self.__dict__["transformation"] = value

    @setter_parms(dict)
    def t_parms(self, value):
        self.__dict__["t_parms"] = value

    @setter_parms(str)
    def spatial_correction(self, value):
        self.__dict__["spatial_correction"] = value

    @setter_parms(dict)
    def sc_parms(self, value):
        self.__dict__["sc_parms"] = value

    @setter_parms(str)
    def normalization(self, value):
        self.__dict__["normalization"] = value

    @setter_parms(dict)
    def n_parms(self, value):
        self.__dict__["n_parms"] = value

    @setter_parms(str)
    def selection(self, value):
        self.__dict__["selection"] = value

    @setter_parms(dict)
    def s_parms(self, value):
        self.__dict__["s_parms"] = value

    @setter_parms(int)
    def validation(self, value):
        self.__dict__["validation"] = value

    @setter_parms(dict)
    def discard(self, value):
        self.__dict__["discard"] = value

    @property
    def json(self):
        """allow to acces a jsonified version of each parameter of an instance
        use : parms.json.ctrl_neg for example"""
        return JsonParms(self.__dict__)

    @property
    def ctrl(self):
        return self.ctrl_neg + self.ctrl_pos

    @property
    def selection_features(self):
        return list(set(f.get("feature", None) for f
                        in self.s_parms.get("str_parms", [])))

    @selection_features.setter
    def selection_features(self, mapper):
        for rule in self.s_parms.get("str_parms", []):
            rule['feature'] = mapper.get(rule['feature'], rule['feature'])

    def __init__(self, json_string=None, parent=None, **kwargs):
        """analysis_type is not used as there are
        only one type of analysis yet"""
        self._parent = parent

        dict_parms = json.loads(json_string) \
            if json_string is not None else kwargs

        self.ctrl_neg = dict_parms.get("ctrl_neg", None)
        self.ctrl_pos = dict_parms.get("ctrl_pos", None)
        self.ctrl_other = dict_parms.get("ctrl_other", None)

        self.features = dict_parms.get("features", None)

        self.transformation = dict_parms.get("transformation", None)
        self.t_parms = dict_parms.get("t_parms", {}) if self.transformation else {}

        self.spatial_correction = dict_parms.get("spatial_correction", None)
        self.sc_parms = dict_parms.get("sc_parms", {}) if self.spatial_correction else {}

        self.normalization = dict_parms.get("normalization", None)
        self.n_parms = dict_parms.get("n_parms", {}) if self.normalization else {}

        self.selection = dict_parms.get("selection", None)
        self.s_parms = dict_parms.get("s_parms", {}) if self.selection else {}

        self.validation = dict_parms.get("validation", -1)

        self.discard = dict_parms.get("discard", {})

    def get(self, item, default=None):
        return self.__dict__.get(item, default)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def keys(self):
        return self.__dict__.keys()

    def __call__(self, **kwargs):
        """add arg afterward"""
        for key, value in kwargs.items():
            if not key.startswith("_") and key in self.__dict__:
                # protect private variable
                setattr(self, key, value)

    def __repr__(self):  # needed for caching
        return str(self.__dict__)

    def dumps(self):
        return json.dumps({k: v for k, v in self.__dict__.items() if not k.startswith('_')})

    def get_parms_of(self, method_type):
        if method_type == "transformation":
            return self.__dict__["t_parms"]
        elif method_type == "spatial_correction":
            return self.__dict__["sc_parms"]
        elif method_type == "normalization":
            return self.__dict__["n_parms"]
        elif method_type == "selection":
            return self.__dict__["s_parms"]

    def __eq__(self, other):
        for key in self.__dict__:
            if key == '_parent':
                continue
            if (key not in other.__dict__ or
                    self.__dict__[key] != other.__dict__[key]):
                return False
        else:
            return True

    def repr_features(self):
        is_dl = 0
        for f in self.features:
            if is_dl is not None:
                try:
                    is_dl = int(f)
                except ValueError:
                    is_dl = None
        if is_dl is not None:
            return DEEP_LEARNING_FEATURES
        return ', '.join(self.features)

    @property
    def other_parms(self):
        return self._parent.other_parms

    def selection_format_excel(self, format1, format2):
        result = {}
        for rule in self.s_parms.get("str_parms", []):
            result[rule['feature']] = []
            op = rule['relative']
            value = rule['value']
            if op == '><':
                result[rule['feature']].append({'type': 'cell',
                                                'criteria': '>',
                                                'value': value,
                                                'format': format1})
                op = '<'
                value = - value
                format_ = format2
            elif op == '<':
                format_ = format2
            else:
                format_ = format1
            result[rule['feature']].append({'type': 'cell',
                                            'criteria': op,
                                            'value': value,
                                            "format": format_})
        return result

    def get_interval_of(self, feature):
        result = Interval()
        for rule in self.s_parms.get('str_parms', []):
            if feature == rule['feature']:
                val = float(rule['value'])
                if rule['relative'] == '><':
                    ts = Interval(AtomicInterval(-inf, -val),
                                  AtomicInterval(val, inf))
                elif rule['relative'] == '>':
                    ts = AtomicInterval(val, inf)
                else:  # rule['relative'] == '<' :
                    ts = AtomicInterval(-inf, val)

                if rule["include"] == 'and':
                    result = Interval(result & ts)
                else:
                    result = Interval(result, ts)
        return result

    # def repr_selection_parms(self, s_parms):
    #     result = ""
    #     if 'str_parms' in s_parms:
    #         result += "Selection rules : "
    #         selection_feature = list(set(rule['feature'] for rule in s_parms['str_parms']))
    #         for feature in selection_feature:
    #             interval = self.get_interval_of(feature)
    #             for atomic in interval:
    #                 first_part = f"{atomic.lower} < " if atomic.lower != -inf else ''
    #                 last_part = f" < {atomic.upper}" if atomic.upper != inf else ''
    #                 result += f"{first_part}value of {feature}{last_part} or "
    #             result = result[:-4] + "\n"
    #     if 'other_parms' in s_parms and s_parms['other_parms'] is not None and \
    #             not all(p is None for p in s_parms['other_parms'].values()):
    #         result += f"Other parms : {s_parms['other_parms']}"
    #     return result

    # def method_df(self):
    #     methods = []
    #     ref = []
    #     for name in ['transformation', 'spatial_correction',
    #                  'normalization', "selection"]:
    #         methods.append(self.__dict__[name])
    #         try:
    #             ref.append((Method(step=name)._get_function(
    #                 self.__dict__[name]
    #             ).__doc__ or '').split('`')[-2])
    #         except IndexError:
    #             ref.append('')
    #     return DataFrame({
    #         "Step": ['Transformation', 'Spatial Correction',
    #                  'Plate Alignment', 'Hit Selection'],
    #         "Method": methods,
    #         "Parameters of method": [
    #             str(self.t_parms).strip(',{[]}'),
    #             str(self.sc_parms).strip(',{[]}'),
    #             str(self.n_parms).strip(',{[]}'),
    #             self.repr_selection_parms(self.s_parms)
    #         ],
    #         "Reference": ref
    #     })

    # def to_html(self, css_class=''):
    #     method_df = self.method_df()
    #     method_col = f"<tr>{''.join([f'<td>{col}</td>' for col in method_df.columns])}</tr>"
    #     method_html = ""
    #     for _, row in method_df.iterrows():
    #         method_html += "<tr>"
    #         for cell in row:
    #             method_html += f"<td>{cell}</td>"
    #         method_html += "</tr>"
    #     table_class = "'avoid-break'"
    #     if css_class:
    #         table_class = table_class[:-1] + " " + css_class + "'"
    #     return (f"<table class={table_class}>"
    #             f"<tr><td>Features</td><td>{self.repr_features()}</td></tr>"
    #             f"<tr><td>Negatives Control</td><td>{', '.join(self.ctrl_neg)}</td></tr>"
    #             f"<tr><td>Positives Control</td><td>{', '.join(self.ctrl_pos)}</td></tr>"
    #             "<tr><td> </td><td> </td></tr>" + method_col + method_html + "</table>")

    def copy(self):
        return copy.deepcopy(self)


class JsonParms(object):
    def __init__(self, dico):
        self.__dict__ = {key: json.dumps(value) for key, value in dico.items()}
