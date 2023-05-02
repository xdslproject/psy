def flatten(provided_list):
    to_ret=[]
    for item in provided_list:
      if isinstance(item, list):
        to_ret.extend(flatten(item))
      else:
        to_ret.append(item)
    return to_ret
