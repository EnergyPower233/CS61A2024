(define (curry-cook formals body) (if (null? formals) body `(lambda (,(car formals)) ,(curry-cook (cdr formals) body))))

(define (curry-consume curry args)
 (if (null? args) curry (curry-consume (curry (car args)) (cdr args))))

(define-macro (switch expr options)
  (switch-to-cond (list 'switch expr options)))


;?
(define (switch-to-cond switch-expr)
  (cons `cond
        (map ;生成列表
              (lambda (option)
               (cons ;组装
               `(equal? ,(car (cdr switch-expr)) ,(car option)) ;此乃 bool
               (cdr option)));把上面的那行和这个组了 是cond的一个情况
             (car (cdr (cdr switch-expr))))));这是所有options
