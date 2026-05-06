import re
from collections import namedtuple

from fplip.basic import config
from fplip.basic.logger import logger
from fplip.basic.supplemental import read

# Try to import Cython-optimized parser
try:
    from plip.structure._pdb_parser import \
        fix_pdbline_str as _fix_pdbline_cython
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


class PDBParser:
    # 类级别的预编译正则表达式和常量，避免每次实例化都重新编译
    _FORBIDDEN_PATTERN = re.compile(r"[^a-zA-Z0-9_]")
    _PDBQT_PATTERN = re.compile(r"(HD|HS|NA|NS|OA|OS|SA)$")
    _PDBQT_MAP = {
        "HD": "H", "HS": "H", "NA": "N",
        "NS": "N", "OA": "O", "OS": "O", "SA": "S"
    }

    def __init__(self, pdbpath, as_string):
        self.as_string = as_string
        self.pdbpath = pdbpath
        self.num_fixed_lines = 0
        self.covlinkage = namedtuple("covlinkage", "id1 chain1 pos1 conf1 id2 chain2 pos2 conf2")
        self.pdb_file_was_corrected = False
        self.proteinmap, self.modres, self.covalent, self.altconformations, self.corrected_pdb = self.parse_pdb()

    def parse_pdb(self):
        """Extracts additional information from PDB files.
        I. When reading in a PDB file, OpenBabel numbers ATOMS and HETATOMS continously.
        In PDB files, TER records are also counted, leading to a different numbering system.
        This functions reads in a PDB file and provides a mapping as a dictionary.
        II. Additionally, it returns a list of modified residues.
        III. Furthermore, covalent linkages between ligands and protein residues/other ligands are identified
        IV. Alternative conformations
        """
        if self.as_string:
            fil = self.pdbpath.rstrip('\n').split('\n')  # Removing trailing newline character
        else:
            f = read(self.pdbpath)
            fil = f.readlines()
            f.close()
        corrected_lines = []
        i, j = 0, 0  # idx and PDB numbering
        d = {}
        modres = set()
        covalent = []
        alt = []
        previous_ter = False

        model_dict = {0: list()}
        # Standard without fixing
        if not config.NOFIX:
            if not config.PLUGIN_MODE:
                lastnum = 0  # Atom numbering (has to be consecutive)
                other_models = False
                # Model 0 stores header and similar additional data
                # or the full file if no MODEL entries exist in the file
                current_model = 0
                for line in fil:
                    corrected_line, newnum = self.fix_pdbline(line, lastnum)
                    if corrected_line is not None:
                        if corrected_line.startswith('MODEL'):
                            # reset atom number when new model is encountered
                            lastnum = 0
                            try:  # Get number of MODEL (1,2,3)
                                model_num = int(corrected_line[10:14])
                                # initialize storage for new model
                                model_dict[model_num] = list()
                                current_model = model_num
                                if model_num > 1:  # MODEL 2,3,4 etc.
                                    other_models = True
                            except ValueError:
                                logger.debug(f'ignoring invalid MODEL entry: {corrected_line}')
                        else:
                            lastnum = newnum
                        model_dict[current_model].append(corrected_line)
                # select model
                try:
                    if other_models:
                        logger.info(f'selecting model {config.MODEL} for analysis')
                    corrected_pdb = ''.join(model_dict[0])
                    self.pdb_file_was_corrected = True
                    corrected_lines = model_dict[0]
                    if current_model > 0:
                        corrected_pdb += ''.join(model_dict[config.MODEL])
                        corrected_lines += model_dict[config.MODEL]
                except KeyError:
                    corrected_pdb = ''.join(model_dict[1])
                    self.pdb_file_was_corrected = True
                    corrected_lines = model_dict[1]
                    config.MODEL = 1
                    logger.warning('invalid model number specified, using first model instead')
            else:
                corrected_pdb = self.pdbpath
                corrected_lines = fil
        else:
            corrected_pdb = self.pdbpath
            corrected_lines = fil

        for line in corrected_lines:
            if line[:6] in ("ATOM  ", "HETATM"):
                # Retrieve alternate conformations
                atomid, location = int(line[6:11]), line[16]
                location = 'A' if location == ' ' else location
                if location != 'A':
                    alt.append(atomid)

                if not previous_ter:
                    i += 1
                    j += 1
                else:
                    i += 1
                    j += 2
                d[i] = j
                previous_ter = False
            # Get covalent linkages between ligands
            elif line[:4] == "LINK":
                covalent.append(self.get_linkage(line))
            # Numbering Changes at TER records
            elif line[:3] == "TER":
                previous_ter = True
            # Get modified residues
            elif line[:6] == "MODRES":
                modres.add(line[12:15].strip())
        return d, modres, covalent, alt, corrected_pdb

    def fix_pdbline(self, pdbline, lastnum):
        """Fix a PDB line if information is missing.

        Uses Cython-optimized implementation if available, otherwise falls back
        to the pure Python implementation.
        """
        # Use Cython version if available
        if _CYTHON_AVAILABLE:
            result, new_num = _fix_pdbline_cython(pdbline, lastnum)
            if result is not None:
                self.num_fixed_lines += 1
            return result, new_num
        
        # Fallback to Python implementation
        return self._fix_pdbline_python(pdbline, lastnum)
    
    def _fix_pdbline_python(self, pdbline, lastnum):
        """Pure Python implementation of fix_pdbline (fallback)."""
        fixed = False
        new_num = 0

        # 保留原始逻辑：只去掉换行符
        pdbline = pdbline.strip('\n')

        # Some MD / Docking tools produce empty lines, leading to segfaults
        if len(pdbline.strip()) == 0:
            self.num_fixed_lines += 1
            return None, lastnum
        if len(pdbline) > 100:  # Should be 80 long
            self.num_fixed_lines += 1
            return None, lastnum

        # TER Entries also have continuing numbering, consider them as well
        if pdbline.startswith('TER'):
            if not pdbline[6:11]:  # pdb files saved from PyMol skip the number in TER entries
                new_num = lastnum
            else:
                new_num = lastnum + 1

        if pdbline.startswith('ATOM'):
            new_num = lastnum + 1
            current_num = int(pdbline[6:11])
            resnum = pdbline[22:27].strip()

            # Invalid residue number
            try:
                int(resnum)
            except ValueError:
                pdbline = pdbline[:22] + '   0 ' + pdbline[27:]
                fixed = True

            # Invalid characters in residue name - 使用预编译正则
            if self._FORBIDDEN_PATTERN.search(pdbline[17:21].strip()):
                pdbline = pdbline[:17] + 'UNK ' + pdbline[21:]
                fixed = True

            if lastnum + 1 != current_num:
                pdbline = pdbline[:6] + (5 - len(str(new_num))) * ' ' + str(new_num) + ' ' + pdbline[12:]
                fixed = True

            # No chain assigned
            if pdbline[21] == ' ':
                pdbline = pdbline[:21] + 'A' + pdbline[22:]
                fixed = True

            if pdbline.endswith('H'):
                self.num_fixed_lines += 1
                return None, lastnum

            # Sometimes, converted PDB structures contain PDBQT atom types. Fix that.
            # 使用预编译正则替代循环
            match = self._PDBQT_PATTERN.search(pdbline.strip())
            if match:
                pdbqt_type = match.group(1)
                pdbline = pdbline.strip()[:-2] + ' ' + self._PDBQT_MAP[pdbqt_type] + '\n'
                self.num_fixed_lines += 1

        if pdbline.startswith('HETATM'):
            new_num = lastnum + 1
            try:
                current_num = int(pdbline[6:11])
            except ValueError:
                current_num = None
                logger.debug(f'invalid HETATM entry: {pdbline}')

            if lastnum + 1 != current_num:
                pdbline = pdbline[:6] + (5 - len(str(new_num))) * ' ' + str(new_num) + ' ' + pdbline[12:]
                fixed = True

            # No chain assigned or number assigned as chain
            if pdbline[21] == ' ':
                pdbline = pdbline[:21] + 'Z' + pdbline[22:]
                fixed = True

            # No residue number assigned
            if pdbline[23:26] == '   ':
                pdbline = pdbline[:23] + '999' + pdbline[26:]
                fixed = True

            # Non-standard Ligand Names
            ligname = pdbline[17:21].strip()
            if len(ligname) > 3:
                pdbline = pdbline[:17] + ligname[:3] + ' ' + pdbline[21:]
                fixed = True

            # 使用预编译正则
            if self._FORBIDDEN_PATTERN.search(ligname.strip()):
                pdbline = pdbline[:17] + 'LIG ' + pdbline[21:]
                fixed = True

            if len(ligname.strip()) == 0:
                pdbline = pdbline[:17] + 'LIG ' + pdbline[21:]
                fixed = True

            if pdbline.endswith('H'):
                self.num_fixed_lines += 1
                return None, lastnum

            # Sometimes, converted PDB structures contain PDBQT atom types. Fix that.
            match = self._PDBQT_PATTERN.search(pdbline.strip())
            if match:
                pdbqt_type = match.group(1)
                pdbline = pdbline.strip()[:-2] + ' ' + self._PDBQT_MAP[pdbqt_type] + ' '
                self.num_fixed_lines += 1

        self.num_fixed_lines += 1 if fixed else 0
        return pdbline + '\n', max(new_num, lastnum)

    def get_linkage(self, line):
        """Get the linkage information from a LINK entry PDB line."""
        conf1, id1, chain1, pos1 = line[16].strip(), line[17:20].strip(), line[21].strip(), int(line[22:26])
        conf2, id2, chain2, pos2 = line[46].strip(), line[47:50].strip(), line[51].strip(), int(line[52:56])
        return self.covlinkage(id1=id1, chain1=chain1, pos1=pos1, conf1=conf1,
                               id2=id2, chain2=chain2, pos2=pos2, conf2=conf2)
