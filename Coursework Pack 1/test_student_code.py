"""
Run this script from the same folder as CW1_solvers.py to generate an HTML file
containing the outputs. Open the file to review the results.
Do not modify this script.
"""

import numpy as np
import time
import re
import importlib
import matplotlib
import matplotlib.pyplot as plt
import errno
import sys



############################################################
case_definitions_dict = {
  
    
   'Q1a Case I': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Command1': {'cmd': 'cw.fixedpoint_with_stopping.__doc__','Output1':'doc_string'},
            },
        'Tests' : {
            'Test 1-I: doc' : {
                'Marks': 2.0,
                'TestObject': 'doc_string',
                'ObjectType': str,
                }
            }
        },
    
   
   'Q1a Case II': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Input1': 'g = lambda x:  1/3*(x**2 -1)',
            'Input2': 'p0 = 1.0',
            'Input4': 'Nmax = 10',
            'Input5': 'TOL = 1e-11',
            'Command1': {'cmd': 'cw.fixedpoint_with_stopping(g,p0,Nmax,TOL)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
        },
        'Tests' : {
            'Test 1-II-1: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type,
                'Marks': 2.0,
                'Test': 'compareToTrueType'
            },
            'Test 1-II-2: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple,
                'Marks': 2.0,
                'Test': 'compareToTrueShape'
            },
            'Test 1-II-3: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray,
                'Marks': 2.0,
                'Test': 'compareToTrueVectorOrShift_Norm',
                'TOL1': 1.0e-10,
                'TOL2': 1.0e-5,
            }
        },
    },
   

   'Q1a Case III': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Input1': 'g = lambda x:  1/3*(x**2 -1)',
            'Input2': 'p0 = 1.0',
            'Input4': 'Nmax = 50',
            'Input5': 'TOL = 1e-11',
            'Command1': {'cmd': 'cw.fixedpoint_with_stopping(g,p0,Nmax,TOL)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
        },
        'Tests' : {
            'Test 1-III-1: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type,
                'Marks': 2.0,
                'Test': 'compareToTrueType'
            },
            'Test 1-III-2: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple,
                'Marks': 2.0,
                'Test': 'compareToTrueShape'
            },
            'Test 1-III-3: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray,
                'Marks': 2.0,
                'Test': 'compareToTrueVectorOrShift_Norm',
                'TOL1': 1.0e-10,
                'TOL2': 1.0e-5,
            }
        },
    },

   'Q1a Case IV': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'matplotlib.pyplot as plt',
            'Import': 'CW1_solvers as cw',
            'Input1': 'g = lambda x:  1/3*(x**2 -1)',
            'Input2': 'p0 = 1.0',
            'Input3': 'Nmax = 10',
            'Input4': 'TOL = 1e-11',
            'Input5': 'p=(3-np.sqrt(13))/2',
#            'Input7': '_,fig,ax = cw.fixedpoint_with_stopping(g,p0,Nmax,TOL,p)',
            'Command1': {'cmd': 'cw.fixedpoint_with_stopping(g,p0,Nmax,TOL,p)', 'Output1': '_,fig,ax'},
            'Command2': {'cmd': 'fig','Output1':'figI'},

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test1-IV: Plot' : {

            'TestObject': 'figI',
            'ObjectType': (matplotlib.figure.Figure),
            }
        }
    },


   'Q1a Case V': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'matplotlib.pyplot as plt',
            'Import': 'CW1_solvers as cw',
            'Input1': 'g = lambda x:  1/3*(x**2 -1)',
            'Input2': 'p0 = 1.0',
            'Input3': 'Nmax = 10',
            'Input4': 'TOL = 1e-11',
            'Input5': 'p=(3-np.sqrt(13))/2',
            'Input6': 'C=2',
            'Input7': 'k=2/3',
#            'Input7': '_,fig,ax = cw.fixedpoint_with_stopping(g,p0,Nmax,TOL,p)',
            'Command1': {'cmd': 'cw.fixedpoint_with_stopping(g,p0,Nmax,TOL,p,C,k)', 'Output1': '_,fig,ax'},
            'Command2': {'cmd': 'fig','Output1':'figII'},

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test1-V: Plot' : {

            'TestObject': 'figII',
            'ObjectType': (matplotlib.figure.Figure),
            }
        }
    },



   'Q1b': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'CW1_solvers as cw',
            'Command1': {'cmd': 'cw.show_answer(cw.q1B_answer, 60)', 'Output1': 'answer'},
            

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test1-VI: Plot' : {

            'TestObject': 'answer',
            'ObjectType': str,
            }
        }
    },
   

   'Q1c': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'CW1_solvers as cw',
            'Command1': {'cmd': 'cw.show_answer(cw.q1C_answer, 60)', 'Output1': 'answer'},
            

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test1-VII: Plot' : {

            'TestObject': 'answer',
            'ObjectType': str,
            }
        }
    },






   'Q2a Case I': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Input1': 'f = lambda x: np.cos(x) - x',
            'Input2': 'df = lambda x: -np.sin(x) - 1',
            'Input3': 'p0 = 0.0',
            'Input4': 'Nmax = 3',
            'Input5': 'TOL = 1e-16',
            'Command1': {'cmd': 'cw.newton_with_stopping(f,df,p0,Nmax,TOL)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
        },
        'Tests' : {
            'Test 2-I-1: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type,
                'Marks': 2.0,
                'Test': 'compareToTrueType'
            },
            'Test 2-I-2: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple,
                'Marks': 2.0,
                'Test': 'compareToTrueShape'
            },
            'Test 2-I-3: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray,
                'Marks': 2.0,
                'Test': 'compareToTrueVectorOrShift_Norm',
                'TOL1': 1.0e-10,
                'TOL2': 1.0e-5,
            }
        },
    },



   'Q2a Case II': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Input1': 'f = lambda x: np.cos(x) - x',
            'Input2': 'df = lambda x: -np.sin(x) - 1',
            'Input3': 'p0 = 0.0',
            'Input4': 'Nmax = 10',
            'Input5': 'TOL = 1e-16',
            'Command1': {'cmd': 'cw.newton_with_stopping(f,df,p0,Nmax,TOL)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
            'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
        },
        'Tests' : {
            'Test 2-II-1: p_array_type' : {
                'TestObject': 'p_array_type',
                'ObjectType': type,
                'Marks': 2.0,
                'Test': 'compareToTrueType'
            },
            'Test 2-II-2: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple,
                'Marks': 2.0,
                'Test': 'compareToTrueShape'
            },
            'Test 2-II-3: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray,
                'Marks': 2.0,
                'Test': 'compareToTrueVectorOrShift_Norm',
                'TOL1': 1.0e-10,
                'TOL2': 1.0e-5,
            }
        },
    },

   'Q2a Case III': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'matplotlib.pyplot as plt',
            'Import': 'CW1_solvers as cw',
            'Input1': 'f = lambda x: np.cos(x) - x',
            'Input2': 'df = lambda x: -np.sin(x) - 1',
            'Input3': 'p0 = 0.0',
            'Input4': 'Nmax = 20',
            'Input5': 'TOL = 1e-16',
            'Input6': 'p = np.float64(0.73908513321516064165531207047)',
#            'Input7': '_,fig,ax = cw.fixedpoint_with_stopping(g,p0,Nmax,TOL,p)',
            'Command1': {'cmd': 'cw.newton_with_stopping(f,df,p0,Nmax,TOL,p)', 'Output1': '_,fig,ax'},
            'Command2': {'cmd': 'fig','Output1':'figIII'},

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test2-III: Plot' : {

            'TestObject': 'figIII',
            'ObjectType': (matplotlib.figure.Figure),
            }
        }
    },



   'Q2b': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'CW1_solvers as cw',
            'Command1': {'cmd': 'cw.show_answer(cw.q2B_answer, 60)', 'Output1': 'answer'},
            

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test2-IV: Plot' : {

            'TestObject': 'answer',
            'ObjectType': str,
            }
        }
    },
   


   'Q3 Case I': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Input1': 'f = lambda x: x**2 - 2',
            'Input2': 'p0 = 1.0',
            'Input3': 'p1 = 2.0',
            'Input4': 'Nmax = 12',
            'Input5': 'TOL = 10**(-6)',
            'Command1': {'cmd': 'cw.secant_with_stopping(f,p0,p1,Nmax,TOL)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
        },
        'Tests' : {
            'Test 3-I-1: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple,
                'Marks': 1.5,
                'Test': 'compareToTrueShape'
            },
            'Test 3-I-2: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray,
                'Marks': 1.5,
                'Test': 'compareToTrueVector_NormGenerous',
                'TOL1': 1.0e-10,
                'TOL2': 1.0e-5,
            }
        },
    },
   

   'Q3 Case II': {
        'GeneralCommands': {
            'Import': 'CW1_solvers as cw',
            'Input1': 'f = lambda x: x**2 - 2',
            'Input2': 'p0 = 1.0',
            'Input3': 'p1 = 2.0',
            'Input4': 'Nmax = 12',
            'Input5': 'TOL = 10**(-16)',
            'Command1': {'cmd': 'cw.secant_with_stopping(f,p0,p1,Nmax,TOL)', 'Output1': 'p_array'},
            'Command2': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
        },
        'Tests' : {
            'Test 3-II-1: p_array_shape': {
                'TestObject': 'p_array_shape',
                'ObjectType': tuple,
                'Marks': 1.5,
                'Test': 'compareToTrueShape'
            },
            'Test 3-II-2: p_array' : {
                'TestObject': 'p_array',
                'ObjectType': np.ndarray,
                'Marks': 1.5,
                'Test': 'compareToTrueVector_NormGenerous',
                'TOL1': 1.0e-10,
                'TOL2': 1.0e-5,
            }
        },
    },






'Q4a Case I': {
    'GeneralCommands': {
        'Import': 'numpy as np',
        'Import': 'matplotlib.pyplot as plt',
        'Import': 'CW1_solvers as cw',
        'Input1': 'f = lambda x: x - np.cos(x)',
        'Input2': 'df = lambda x: 1 + np.sin(x)',
        'Input3': 'p0_newton = 0.0',
        'Input4': 'p0_sec = 0.0',
        'Input5': 'p1_sec=2.0',
        'Input6': 'Nmax = 20',
        'Input7': 'pe = np.float64(0.73908513321516064165531207047)',
        'Input8': 'TOL = 1e-16',
        'Input9': 'fig,ax = cw.plot_convergence(pe,f,df,p0_newton,p0_sec,p1_sec,Nmax,TOL)',
        'Command1': {'cmd': 'fig','Output1':'fig4I'},
        },
    'Tests' : {
        'Test4-I: Plot' : {

            'TestObject': 'fig4I',
            'ObjectType': (matplotlib.figure.Figure),
            }
        }
    },


   'Q4b': {
        'GeneralCommands': {
            'Import': 'numpy as np',
            'Import': 'CW1_solvers as cw',
            'Command1': {'cmd': 'cw.show_answer(cw.q4B_answer, 80)', 'Output1': 'answer'},
            

#            'Command1': {'cmd': 'fig','Output2':'fig4II'},
        },
    'Tests' : {
        'Test4-2: Plot' : {

            'TestObject': 'answer',
            'ObjectType': str,
            }
        }
    },
   
   
    'Q5a Case I': {
         'GeneralCommands': {
             'Import': 'CW1_solvers as cw',
             'Command1': {'cmd': 'cw.scaled_pivoting.__doc__','Output1':'doc_string'},
             },
         'Tests' : {
             'Test 5-I: doc' : {
                 'Marks': 2.0,
                 'TestObject': 'doc_string',
                 'ObjectType': str,
                 }
             }
         },




    'Q5a Case II': {
     'GeneralCommands': {
         'Import': 'CW1_solvers as cw',
         'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]], dtype=float)',
         'Input2': 'b = np.array([[7],[6],[4]], dtype=float)',
         'Input3': 'n=3',
         'Input4': 'm1=1',
         'Input5': 'm2=2',
         'Command1': {'cmd': 'cw.scaled_pivoting(A,b,m1)', 'Output1': 'M1,perm1'},
         'Command2': {'cmd': 'M1','Output1':'M1'},
         'Command3': {'cmd': 'perm1','Output1':'perm1'},
         'Command4': {'cmd': 'cw.scaled_pivoting(A,b,m2)', 'Output1': 'M2,perm2'},
         'Command5': {'cmd': 'M2','Output1':'M2'},
         'Command6': {'cmd': 'perm2','Output1':'perm2'},
         },
     'Tests' : {
         'Test 5-II-1' : {
             'TestObject': 'M1',
             'ObjectType': np.ndarray,
             'Marks': 0.5,
             'Test': 'TrueMatrixEles',
             'Tol': 1.0e-10,
             },
         'Test 5-II-2' : {
             'TestObject': 'M2',
             'ObjectType': np.ndarray,
             'Marks': 0.5,
             'Test': 'TrueMatrixEles',
             'Tol': 1.0e-10,
             },
         'Test 5-II-3' : {
             'TestObject': 'perm1',
             'ObjectType': np.ndarray,
             'Marks': 0.5,
             'Test': 'compareToTrueVector_NormGenerous',
             'Tol': 1.0e-10,
             },
         'Test 5-II-4' : {
             'TestObject': 'perm2',
             'ObjectType': np.ndarray,
             'Marks': 0.5,
             'Test': 'compareToTrueVector_NormGenerous',
             'Tol': 1.0e-10,
             },         
         },
     },
    

    

    'Q5b Case I': {
     'GeneralCommands': {
         'Import': 'CW1_solvers as cw',
         'Input1': 'A = np.array([[1,-5,1],[10,0.0,20],[5,10,-1]], dtype=float)',
         'Input2': 'b = np.array([[7],[6],[4]], dtype=float)',
         'Command1': {'cmd': 'cw.sp_solve(A,b)', 'Output1': 'x'},
         },
     'Tests' : {
         'Test 5-III-1' : {
             'TestObject': 'x',
             'ObjectType': np.ndarray,
             'Marks': 0.5,
             'Test': 'TrueMatrixEles',
             'Tol': 1.0e-10,
             },
         },
     },
    
    
     
    
    
    'Q5c': {
         'GeneralCommands': {
             'Import': 'numpy as np',
             'Import': 'CW1_solvers as cw',
             'Command1': {'cmd': 'cw.show_answer(cw.q5C_answer, 60)', 'Output1': 'answer'},
             

 #            'Command1': {'cmd': 'fig','Output2':'fig4II'},
         },
     'Tests' : {
         'Test5-IV: Plot' : {

             'TestObject': 'answer',
             'ObjectType': str,
             }
         }
     },



    
   # 'Q3 Case I': {
   #      'GeneralCommands': {
   #          'Import': 'rootsolvers as rs',
   #          'Input1': 'f = lambda x: x**3 + x**2 - 2*x - 2',
   #          'Input2': 'c = 1/12',
   #          'Input3': 'p0 = 1',
   #          'Input4': 'Nmax = 10',
   #          'Command1': {'cmd': 'rs.fp_iteration(f,c,p0,Nmax)', 'Output1': 'p_array'},
   #          'Command2': {'cmd': 'type(p_array)', 'Output1': 'p_array_type'},
   #          'Command3': {'cmd': 'np.shape(p_array)','Output1': 'p_array_shape'},
   #          'Command4': {'cmd': 'rs.fp_iteration.__doc__','Output1':'doc_string'},
   #          },
   #      'Tests' : {
   #          'Test: help(rs.fp_iteration)' : {
   #              'Marks': 2.,
   #              'Test': 'Manual',
   #              'TestObject': 'doc_string',
   #              'ObjectType': str,
   #              }
   #          }
   #      },

    # 'Comments': {
    #     'GeneralCommands': {
    #         'Command': {'cmd': 'convert_file_to_html(directory+"/Lagrange.py")','Output1': 'file_as_string'}
    #         },
    #     'Tests' : {
    #         'Test8' : {
    #             'Marks': 10.,
    #             'Test': 'Manual',
    #             'ShowModel': True,
    #             'TestObject': 'file_as_string'
    #             }
    #         }
    #     }

    }

############################################################
def run_test_case(dict_name,case_name,student_command_window):
    
    """
    Runs test case case_name from the dict_name
    It is assumed all files to import are in the current directory
    """

    # Change plt.show() - to avoid pauses when running the code
    plt.show2 = plt.show
    plt.show = lambda x=1: None
    
    student_command_window[case_name] = "<tt>"
    
    key_list = list(dict_name[case_name]["GeneralCommands"].keys())
    
    glob_dict = globals()
    loc_dict = {}


    #Import modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            cmd = "import "+dict_name[case_name]["GeneralCommands"].get(key)

            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:                
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Variable set up (based on Inputs in dictionary)
    for key in key_list:
        if bool(re.search("^Input",key)):
            cmd = dict_name[case_name]["GeneralCommands"].get(key)
            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Initialise the output dictionary
    dict_name[case_name]["Outputs"] = {}
    
    #Run the set of commands
    
    for key in key_list:
            if bool(re.search("^Command",key)):
                #Set up the outputs
                command_key_list = list(dict_name[case_name]["GeneralCommands"][key].keys())
                Outputs = ""
                for cmd_key in command_key_list:
                    if bool(re.search("^Output",cmd_key)):
                        Outputs = Outputs + dict_name[case_name]["GeneralCommands"][key].get(cmd_key) + ", "

                if len(Outputs) >= 3:
                    Outputs = Outputs[0:len(Outputs)-2]

                cmd = Outputs + " = " + dict_name[case_name]["GeneralCommands"][key].get("cmd")

                student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
                try:
                    exec(cmd,glob_dict,loc_dict)
                        
                    #Append the outputs to the Outputs section of the case dictionary
                    for cmd_key in command_key_list:
                        if bool(re.search("^Output",cmd_key)):
                            output_name = dict_name[case_name]["GeneralCommands"][key].get(cmd_key)
                            dict_name[case_name]["Outputs"][output_name] = loc_dict.get(output_name)
                except Exception as e:
                    student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    student_command_window[case_name] = student_command_window[case_name]+'</tt>'

    #Clean up all newly added modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            if str.split(dict_name[case_name]["GeneralCommands"].get(key))[0] in sys.modules.keys(): 
                del sys.modules[str.split(dict_name[case_name]["GeneralCommands"].get(key))[0]] 
    
    # Change back plt.show()
    plt.show = plt.show2

    # Close all open figures
    plt.close('all')

############################################################

def create_html_of_outputs(student_case_dict,cmd_window):
    

    with open('StudentCodeTestOutput.html','w') as file:
        file.writelines('\n'.join(["<!DOCTYPE html>","<html>"]))
        file.write('<head> \n')
        file.write('Output from Code tests')
        file.write('</head> \n')

        file.write('<body> \n')

        case_keys = student_case_dict.keys()

        for case_name in case_keys:
            print(case_name)  
            if case_name == 'Q4 Case I':
                print(case_name)    

            file.write('<p> <b>Case: '+case_name+'</b><br></p>\n')
            
            #Output the commands run
            file.write('<p style="margin-left:30px;"> <u>Commands Run:</u> </p>\n')
            file.write('<p style="margin-left:60px;"<tt>'+cmd_window[case_name]+'</tt></p>')
            
            key_list = list(student_case_dict[case_name]["Tests"].keys())

            for key in key_list:
                if bool(re.search("^Test",key)):
                    file.write('<pre><p style="margin-left:30px;"><u>'+key+'</u><br></p></pre>\n')
                    test_object_key = student_case_dict[case_name]["Tests"][key].get("TestObject")

                    if "Outputs" in student_case_dict[case_name]:
                            
                        student_output = student_case_dict[case_name]["Outputs"].get(test_object_key)

                        file.write('<p style="margin-left:60px;"> Student Output: </p>\n')
                        

                        student_output_type = type(student_output)
                        required_output_type = student_case_dict[case_name]["Tests"][key].get("ObjectType")
                        
                        if not isinstance(student_output,required_output_type):
                            required_string = str(required_output_type).replace("<","&lt")
                            required_string = required_string.replace(">","&gt")
                            received_string = str(student_output_type).replace("<","&lt")
                            received_string = received_string.replace(">","&gt")
                            warn = "requires (one of) <tt>"+required_string+"</tt> received <tt>"+received_string+"</tt>"
                            file.write("<p style=\"margin-left:60px;\"><span style=\"color:red\">Warning</span>: Student output is of incorrect type, "+warn+"</p>\n")

                        file.write('<pre><p style="margin-left:60px;">'+ test_object_key + ' = </p></pre>')
                            
                        if isinstance(student_output,matplotlib.figure.Figure):
                            student_output.savefig(test_object_key+".png",bbox_inches = "tight")
                                
                            file.write('<p style="margin-left:90px;"><img src="'+test_object_key+".png\" height=\"400\" width=\"600\"></p><br><br>\n")
                        else:
                            student_output = str(student_output)
                        
                            # Escape HTML special chars
                            student_output = student_output.replace("&", "&amp;")
                            student_output = student_output.replace("<", "&lt;")
                            student_output = student_output.replace(">", "&gt;")
                        
                            # Write inside <pre> so \n becomes real line breaks
                            file.write(
                                '<pre style="margin-left:90px; white-space:pre-wrap;">'
                                + student_output +
                                '</pre>\n'
                            )

                            '''
                            student_output = str(student_output)
                            
                            #if isinstance(student_output,str):
                            student_output = student_output.replace("<","&lt")
                            student_output = student_output.replace(">","&gt")
                            student_output = student_output.replace('\n','<br>')
                            #student_output = '<pre> '+student_output+' </pre>'
                            
                            file.write('<pre><p style="margin-left:90px;">'+ student_output +'</p></pre>')
                            '''
                    else:
                        file.write('<p style="margin-left:60px;"> Student Output: None </p>\n')

                
                        
        file.write('</body> \n')
        file.write('</html> \n')


############################################################
#Test the code and output

case_keys = case_definitions_dict.keys()

student_command_window = {}

for case_name in case_keys:

    print("      Running ",case_name)
    #Run the student code and store output
    run_test_case(case_definitions_dict,case_name,student_command_window)

create_html_of_outputs(case_definitions_dict,student_command_window)
print("      Created file StudentCodeTestOutput.html")
print("          (open it in a web browser)")


