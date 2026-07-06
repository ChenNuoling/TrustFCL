import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


def clamp_img(x, x_orig, epsilon, clip_min=0.0, clip_max=1.0):
    """将图像约束到 epsilon-ball 和有效像素范围内"""
    x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
    x = torch.clamp(x, clip_min, clip_max)
    return x


def project_l2(x, x_orig, epsilon):
    """L2 投影到 epsilon-ball"""
    diff = x - x_orig
    batch_size = diff.size(0)
    flat_diff = diff.view(batch_size, -1)
    norms = torch.norm(flat_diff, dim=1)
    
    # 找出需要投影的样本
    mask = norms > epsilon
    
    if mask.any():
        # 对需要投影的样本进行 L2 投影
        scale = epsilon / norms[mask]
        for i in range(batch_size):
            if mask[i]:
                diff[i] = diff[i] * (scale[i].item() / norms[i].item())
    
    return x_orig + diff


def dlr_loss(logits, y, targeted=False):
    """Difference of Logits Ratio Loss"""
    z = logits
    z_y = z[range(len(z)), y]
    z_sorted, _ = torch.sort(z, dim=1, descending=True)
    z_pi1 = z_sorted[:, 0]
    z_pi2 = z_sorted[:, 1]
    z_pi3 = z_sorted[:, 2]
    loss = -(z_y - z_pi2) / (z_pi1 - z_pi3 + 1e-12)
    if targeted:
        return -loss
    return loss.mean()


class Attack:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = args.device
    
    def fgsm(self, x, y, task_id=None):
        """Fast Gradient Sign Method (FGSM)"""
        training_mode = self.model.training
        self.model.eval()
        x_adv = x.clone().detach().requires_grad_(True)
        outputs = self.model(x_adv, task_id)
        loss = F.cross_entropy(outputs, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + self.args.fgsm_eps * grad.sign()
        self.model.train(training_mode)
        return torch.clamp(x_adv, 0, 1).detach()
    
    def pgd(self, x, y, task_id=None, steps=None):
        """Projected Gradient Descent (PGD)"""
        training_mode = self.model.training
        self.model.eval()
        
        _steps = steps if steps is not None else self.args.pgd_test
        
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-self.args.pgd_eps, self.args.pgd_eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        
        for _ in range(_steps):
            x_adv.requires_grad_(True)
            outputs = self.model(x_adv, task_id)
            loss = F.cross_entropy(outputs, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + self.args.pgd_alpha * grad.sign()
                delta = torch.clamp(x_adv - x, -self.args.pgd_eps, self.args.pgd_eps)
                x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        self.model.train(training_mode)
        return x_adv
    
    def _compute_jacobian(self, x, task_id, num_classes):
        """计算前向导数 (Jacobian Matrix)"""
        x_var = x.clone().detach().requires_grad_(True)
        output = self.model(x_var, task_id)
        jacobian = []
        for i in range(num_classes):
            if x_var.grad is not None:
                x_var.grad.zero_()
            output[0, i].backward(retain_graph=True)
            jacobian.append(x_var.grad.clone().detach())
        return torch.stack(jacobian).squeeze(1)
    
    def jsma(self, x, y, task_id=None):
        """Jacobian-based Saliency Map Attack (JSMA)"""
        training_mode = self.model.training
        self.model.eval()
        
        max_distortion = getattr(self.args, 'jsma_max_distortion', 0.1)
        theta = getattr(self.args, 'jsma_theta', 1.0)
        
        x_adv = x.clone().detach()
        batch_size = x.shape[0]
        num_classes = self.args.num_classes // self.args.num_tasks  # 每任务的类别数
        
        for b in range(batch_size):
            img = x[b:b+1].clone().detach()
            target_label = y[b].item()
            c, h, w = img.shape[1:]
            num_features = c * h * w
            max_iters = int(num_features * max_distortion)
            search_space = torch.ones(img.shape[1:], device=self.device).bool()
            
            output = self.model(img, task_id)
            current_pred = output.argmax(dim=1).item()
            
            loop_i = 0
            while current_pred != target_label and loop_i < max_iters:
                jacobian = self._compute_jacobian(img, task_id, num_classes)
                
                alpha = jacobian[target_label]
                beta = torch.sum(jacobian, dim=0) - alpha
                
                mask1 = (alpha > 0)
                mask2 = (beta < 0)
                combined_mask = mask1 & mask2 & search_space
                
                saliency = -alpha * beta * combined_mask.float()
                
                if saliency.max() <= 0:
                    break
                
                idx = torch.argmax(saliency.view(-1)).item()
                p_c = idx // (h * w)
                p_h = (idx % (h * w)) // w
                p_w = idx % w
                
                new_val = img[0, p_c, p_h, p_w] + theta
                img[0, p_c, p_h, p_w] = torch.clamp(new_val, 0, 1)
                search_space[p_c, p_h, p_w] = False
                
                output = self.model(img, task_id)
                current_pred = output.argmax(dim=1).item()
                loop_i += 1
            
            x_adv[b] = img.squeeze(0)
        
        self.model.train(training_mode)
        return x_adv
    
    def deepfool(self, x, y, task_id=None):
        """DeepFool Attack"""
        training_mode = self.model.training
        self.model.eval()
        
        overshoot = getattr(self.args, 'deepfool_overshoot', 0.02)
        max_iter = getattr(self.args, 'deepfool_max_iter', 50)
        num_classes = getattr(self.args, 'deepfool_num_classes', 10)
        
        x_adv = x.clone().detach()
        batch_size = x.shape[0]
        
        for b in range(batch_size):
            img = x[b:b+1].clone().detach()
            img_var = img.clone().detach().requires_grad_(True)
            
            output = self.model(img_var, task_id)
            f_image = output.data.cpu().numpy().flatten()
            I = f_image.argsort()[::-1][:num_classes]
            label = I[0]
            
            input_shape = img.cpu().numpy().shape
            r_tot = np.zeros(input_shape)
            
            loop_i = 0
            x_var = img.clone().detach().requires_grad_(True)
            fs = self.model(x_var, task_id)
            k_i = label
            
            while k_i == label and loop_i < max_iter:
                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)
                grad_orig = x_var.grad.data.cpu().numpy().copy()
                
                for k in range(1, num_classes):
                    if x_var.grad is not None:
                        x_var.grad.zero_()
                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = x_var.grad.data.cpu().numpy().copy()
                    
                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
                    pert_k = abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-8)
                    
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                
                r_i = (pert + 1e-4) * w / (np.linalg.norm(w) + 1e-8)
                r_tot = np.float32(r_tot + r_i)
                
                pert_image = img + (1 + overshoot) * torch.from_numpy(r_tot).to(self.device)
                x_var = pert_image.clone().detach().requires_grad_(True)
                fs = self.model(x_var, task_id)
                k_i = np.argmax(fs.data.cpu().numpy().flatten())
                loop_i += 1
            
            r_tot = (1 + overshoot) * r_tot
            x_adv[b] = torch.clamp(img + torch.from_numpy(r_tot).to(self.device), 0, 1).squeeze(0)
        
        self.model.train(training_mode)
        return x_adv
    
    def apgd(self, x, y, task_id=None):
        """Auto-PGD Attack"""
        training_mode = self.model.training
        self.model.eval()
        
        eps = getattr(self.args, 'pgd_eps', 0.031)
        n_iter = getattr(self.args, 'apgd_iter', 100)
        norm = getattr(self.args, 'attack_norm', 'linf')
        
        batch_size = x.size(0)
        
        delta = torch.empty_like(x).uniform_(-eps, eps)
        x_adv = clamp_img(x + delta, x, eps).detach().requires_grad_(True)
        
        best_x_adv = x_adv.clone()
        best_loss = torch.full((batch_size,), -float('inf'), device=self.device)
        
        step_size = 2.0 * eps
        prev_grad = torch.zeros_like(x_adv)
        
        for iteration in range(n_iter):
            logits = self.model(x_adv, task_id)
            loss = F.cross_entropy(logits, y, reduction='none')
            
            self.model.zero_grad()
            loss.sum().backward()
            
            if x_adv.grad is None:
                continue
                
            grad = x_adv.grad.data
            
            grad = grad + 0.75 * prev_grad
            prev_grad = grad
            
            with torch.no_grad():
                if norm == 'linf':
                    x_adv = x_adv + step_size * grad.sign()
                    x_adv = clamp_img(x_adv, x, eps)
                else:
                    x_adv = x_adv + step_size * grad / (grad.view(batch_size, -1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
                    x_adv = project_l2(x_adv, x, eps)
            
            x_adv = x_adv.detach().requires_grad_(True)
            
            current_loss = loss.detach()
            better = current_loss > best_loss
            best_loss[better] = current_loss[better]
            best_x_adv[better] = x_adv[better].clone()
        
        self.model.train(training_mode)
        return torch.clamp(best_x_adv, 0, 1).detach()
    
    def fab(self, x, y, task_id=None):
        """Fast Adaptive Boundary Attack"""
        training_mode = self.model.training
        self.model.eval()
        
        eps = getattr(self.args, 'pgd_eps', 0.031)
        n_iter = getattr(self.args, 'fab_iter', 50)
        n_targets = getattr(self.args, 'fab_n_targets', 9)
        norm = getattr(self.args, 'attack_norm', 'linf')
        
        batch_size = x.size(0)
        x_adv = x.clone().detach()
        
        with torch.no_grad():
            logits = self.model(x, task_id)
            topk_indices = logits.argsort(dim=1, descending=True)
        
        for target_idx in range(1, min(n_targets + 1, logits.size(1))):
            target = topk_indices[:, target_idx]
            
            with torch.no_grad():
                delta = torch.randn_like(x) * 0.001
                x_adv = clamp_img(x + delta, x, eps)
            
            for iteration in range(n_iter):
                x_adv = x_adv.clone().detach().requires_grad_(True)
                logits = self.model(x_adv, task_id)
                
                f_y = logits[range(batch_size), y]
                f_t = logits[range(batch_size), target]
                loss = (f_t - f_y).sum()
                
                self.model.zero_grad()
                loss.backward()
                
                if x_adv.grad is None:
                    continue
                    
                grad = x_adv.grad.data
                
                with torch.no_grad():
                    if norm == 'linf':
                        step = grad.sign()
                    else:
                        step = grad / (grad.view(batch_size, -1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
                    
                    x_adv = x_adv + 0.1 * step
                    x_adv = clamp_img(x_adv, x, eps)
        
        self.model.train(training_mode)
        return torch.clamp(x_adv, 0, 1).detach()
    
    def square(self, x, y, task_id=None):
        """Square Attack - 黑盒攻击"""
        training_mode = self.model.training
        self.model.eval()
        
        eps = getattr(self.args, 'pgd_eps', 0.031)
        n_queries = getattr(self.args, 'square_n_queries', 1000)
        norm = getattr(self.args, 'attack_norm', 'linf')
        
        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)
        
        best_x_adv = x.clone()
        best_loss = torch.full((batch_size,), float('inf'), device=self.device)
        
        for query in range(n_queries):
            square_size = max(1, min(h, w) // 4)
            h_start = np.random.randint(0, max(1, h - square_size))
            w_start = np.random.randint(0, max(1, w - square_size))
            
            new_delta = torch.zeros_like(x)
            if norm == 'linf':
                perturbation = (np.random.rand() * 2 - 1) * eps
                new_delta[:, :, h_start:h_start+square_size, w_start:w_start+square_size] = perturbation
            else:
                perturbation = torch.randn(1, 1, square_size, square_size, device=self.device)
                perturbation = perturbation / (perturbation.norm() + 1e-12) * eps
                new_delta[:, :, h_start:h_start+square_size, w_start:w_start+square_size] = perturbation
            
            x_try = clamp_img(x + new_delta, x, eps)
            
            with torch.no_grad():
                logits = self.model(x_try, task_id)
                loss = F.cross_entropy(logits, y, reduction='none')
            
            better = loss < best_loss
            best_loss[better] = loss[better]
            best_x_adv[better] = x_try[better].clone()
        
        self.model.train(training_mode)
        return torch.clamp(best_x_adv, 0, 1).detach()
    
    def autoattack(self, x, y, task_id=None):
        """AutoAttack - 集成攻击"""
        training_mode = self.model.training
        self.model.eval()
        
        eps = getattr(self.args, 'pgd_eps', 0.031)
        n_iter = getattr(self.args, 'apgd_iter', 50)
        n_queries = getattr(self.args, 'square_n_queries', 500)
        
        batch_size = x.size(0)
        best_x_adv = x.clone()
        any_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # APGD-CE
        x_adv = self.apgd(x, y, task_id)
        with torch.no_grad():
            logits = self.model(x_adv, task_id)
            pred = logits.argmax(dim=1)
            success = (pred != y)
            best_x_adv[success & ~any_success] = x_adv[success & ~any_success]
            any_success = any_success | success
        
        # FAB
        x_adv = self.fab(x, y, task_id)
        with torch.no_grad():
            logits = self.model(x_adv, task_id)
            pred = logits.argmax(dim=1)
            success = (pred != y)
            best_x_adv[success & ~any_success] = x_adv[success & ~any_success]
            any_success = any_success | success
        
        # Square
        x_adv = self.square(x, y, task_id)
        with torch.no_grad():
            logits = self.model(x_adv, task_id)
            pred = logits.argmax(dim=1)
            success = (pred != y)
            best_x_adv[success & ~any_success] = x_adv[success & ~any_success]
        
        self.model.train(training_mode)
        return torch.clamp(best_x_adv, 0, 1).detach()
    
    def attack(self, x, y, task_id=None, attack_name=None):
        """统一攻击入口"""
        if attack_name is None:
            raise ValueError("attack_name must be specified")
        
        if attack_name == 'pgd':
            return self.pgd(x, y, task_id)
        elif attack_name == 'fgsm':
            return self.fgsm(x, y, task_id)
        elif attack_name == 'jsma':
            return self.jsma(x, y, task_id)
        elif attack_name == 'deepfool':
            return self.deepfool(x, y, task_id)
        elif attack_name == 'apgd':
            return self.apgd(x, y, task_id)
        elif attack_name == 'fab':
            return self.fab(x, y, task_id)
        elif attack_name == 'square':
            return self.square(x, y, task_id)
        elif attack_name == 'autoattack':
            return self.autoattack(x, y, task_id)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")