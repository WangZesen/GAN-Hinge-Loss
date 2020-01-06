import argparse
from train import train
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, choices = ['mnist'], 
	default = 'mnist', help = 'choice of dataset')
parser.add_argument('--learning_rate', type = float, default = 1e-4,
	help = 'initial learning rate')
parser.add_argument('--noise_dim', type = int, default = 32,
	help = 'dimension of random noise')
parser.add_argument('--batch_size', type = int, default = 64,
	help = '# of data in one batch')

args = parser.parse_args()

def apply_configure(args):
	# GPU Mode
	gpus = []
	for i in range(16):
		try:
			out = subprocess.check_output(['nvidia-smi', '-i', str(i)]).decode('utf-8').split('\n')
			if out[15].find('No running') != -1:
				gpus.append(str(i))
				break
		except:
			break
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

if __name__ == '__main__':

	# GPU Configuration
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	# Start Training
	train(args)