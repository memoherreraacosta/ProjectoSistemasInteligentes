#------------------------------------------------------------------------------------------------------------------
#   Bayesian network for the burglary alarm problem.
#------------------------------------------------------------------------------------------------------------------

import pyAgrum as gum

# Initialize model
model = gum.BayesNet('AdmisionExam')

# Define Exam Level variable
exam_level_var = gum.LabelizedVariable('Exam_Level', 'Exam_Level', 2)
exam_level_var.changeLabel(0, 'F') # Difficult
exam_level_var.changeLabel(1, 'T') # Easy

# Define Iq Level variable
iq_level_var = gum.LabelizedVariable('Iq_level', 'Iq_level', 2)
iq_level_var.changeLabel(0, 'F') # Low
iq_level_var.changeLabel(1, 'T') # High

# Define Apt Score variable
apt_score_var = gum.LabelizedVariable('Apt_score', 'Apt_score', 2)
apt_score_var.changeLabel(0, 'F')
apt_score_var.changeLabel(1, 'T')

# Define Marks variable
marks_var = gum.LabelizedVariable('Marks', 'Marks', 2)
marks_var.changeLabel(0, 'F')
marks_var.changeLabel(1, 'T')

# Define Admission variable
admin_var = gum.LabelizedVariable('Admission', 'Admission', 2)
admin_var.changeLabel(0, 'F')
admin_var.changeLabel(1, 'T')

# Build model
exam_level = model.add(exam_level_var)
iq_level = model.add(iq_level_var)
apt_score = model.add(apt_score_var)
marks = model.add(marks_var)
admin = model.add(admin_var)

model.addArc(exam_level, marks)
model.addArc(iq_level, marks)
model.addArc(iq_level, apt_score)
model.addArc(marks, admin)

# Define conditional probabilities
# Exam_Level, Iq_level, Apt_score, Marks, Admission


# Exam Level
model.cpt(exam_level)[:] = [0.7, 0.3]

# IQ Level
model.cpt(iq_level)[:] = [0.8, 0.2]

# Marks
model.cpt(marks)[{'Iq_level':'T', 'Exam_Level': 'T'}] = [0.8, 0.2] 
model.cpt(marks)[{'Iq_level':'T', 'Exam_Level': 'F'}] = [0.5, 0.5]
model.cpt(marks)[{'Iq_level':'F', 'Exam_Level': 'T'}] = [0.9, 0.1]
model.cpt(marks)[{'Iq_level':'F', 'Exam_Level': 'F'}] = [0.6, 0.4]

# Apt Score
model.cpt(apt_score)[{'Iq_level':'T'}] = [0.4, 0.6] 
model.cpt(apt_score)[{'Iq_level':'F'}] = [0.75, 0.25]

# Admin
model.cpt(admin)[{'Marks': 'T'}] = [0.9, 0.1]
model.cpt(admin)[{'Marks': 'F'}] = [0.6, 0.4]

# Make inference
ie=gum.LazyPropagation(model)
ie.makeInference()

# P1
print('\nDistribuicion conjunta del modelo:')
ie.setEvidence({})
ie.addJointTarget({admin, apt_score, marks, iq_level, exam_level})
print(ie.jointPosterior({admin, apt_score, marks, iq_level, exam_level}))


# P2
print('\n¿Cual es la probabilidad de tener un'
      'alto IQ dado que el alumno fue aceptado '
      'en la universidad?')
ie.setEvidence({'Admission': 'T'})
ie.addTarget(iq_level)
print(ie.posterior(iq_level))


# P3
print('\n¿Cual es la probabilidad de ser aceptado?')
ie.setEvidence({})
ie.addTarget(admin)
print(ie.posterior(admin))


# P4
print('\n¿Cual es la probabilidad de ser aceptado '
      'si se tiene un puntaje bajo de aptitud?')
ie.setEvidence({'Apt_score': 'F'})
ie.addTarget(admin)
print(ie.posterior(admin))
