class KwList:
    @staticmethod
    def _kwlist_unique_elements(kw_list:list) -> list:
        return [element for element in set(kw_list)]
