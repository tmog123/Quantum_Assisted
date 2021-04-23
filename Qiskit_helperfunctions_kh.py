from qiskit.circuit.library import EfficientSU2
import qiskit.tools.jupyter # import handy jupyter tools for viewing backend details and monitoring job status
from qiskit import ClassicalRegister

#Same as jon
from qiskit import IBMQ
from qiskit import QuantumCircuit, execute, result, QuantumRegister
from qiskit.providers.aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators.primitive_ops import CircuitOp
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
from qiskit.aqua.operators.state_fns import CircuitStateFn
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn


#load IBMQ account
# This will throw an erorr if IMBQ account is not saved, consult qiskit docs for help
IBMQ.load_account() 

def measurement_error_mitigator(systemsize, sim, qc = "ibmq_rome",
    shots = 8192):
    if sim == "noisy":
        backend = Aer.get_backend('qasm_simulator')
        provider = IBMQ.get_provider(hub='ibm-q-nus',group='default',
        project='reservations') #Change this line accordingly
        noisebackend = provider.get_backend(qc)
        print("Noise provider backend: ", noisebackend)
        device = noisebackend
        coupling_map = device.configuration().coupling_map
        #noise_model = NoiseModel.from_backend(device,gate_error=False,readout_error=False,thermal_relaxation=True)
        noise_model = NoiseModel.from_backend(device)

        qr = QuantumRegister(systemsize)
        meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
        print('Calibrating POVM Matrix')
        job = execute(meas_calibs,backend=backend,shots=shots,noise_model=noise_model,coupling_map=coupling_map)
        cal_results = job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
        meas_filter = meas_fitter.filter
    elif sim == "real":
        provider = IBMQ.get_provider(hub='ibm-q-nus', group='default', project='reservations')
        backend = provider.get_backend("ibmq_rome")

        qr = QuantumRegister(systemsize)
        meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
        print('Calibrating POVM Matrix')
        job = execute(meas_calibs,backend=backend,shots=shots)
        job_monitor(job, interval = 2)
        cal_results = job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
        meas_filter = meas_fitter.filter
    print("Provider backend: ", backend)
    return meas_filter



# Still doing!