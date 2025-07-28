import sys
import os


def replace_package(input_urdf, output_urdf, pkg_root):
    """
    URDFのpackage`package`を絶対pathに書き換える
    """
    with open(input_urdf, mode="r", encoding="utf-8") as fin:
        urdf = fin.read().rstrip("\n")
        
    replace_urdf = urdf.replace(
        "package://openarm_description/",
        f"{pkg_root}/"
    )

    with open(output_urdf, mode="w", encoding="utf-8") as fout:
        fout.write(replace_urdf+"\n")
